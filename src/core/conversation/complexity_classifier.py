"""
Message Complexity Classifier — Routes messages to fast or deep pipeline paths.

WHAT: Classifies incoming messages into three complexity tiers:
      - SIMPLE: greetings, check-ins, short questions, casual banter
      - COMPLEX: emotionally rich, contextually dense, multi-topic, deep conversation
      - ACTION: messages requiring tool execution (weather, email, calendar, etc.)

WHY:  Simple messages don't need the full 24-source context assembly or graph
      traversal. By routing them to a lightweight path, we can dramatically reduce
      response latency for ~40-60% of messages without sacrificing quality on the
      ones that need depth.

HOW:  Two-tier classification:
      1. Fast keyword/heuristic pre-filter (< 1ms) catches obvious cases
      2. If ambiguous, falls through to the full pipeline (conservative default)
      No LLM call needed — this must be fast to justify its existence.

Feature flag: COMPANION_FAST_PATH_ENABLED (default: false)
"""

import os
import re
import logging
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

FAST_PATH_ENABLED = os.environ.get(
    'COMPANION_FAST_PATH_ENABLED', 'true'
).lower() == 'true'


class MessageComplexity(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    ACTION = "action"


@dataclass
class ClassificationResult:
    """Result of message complexity classification."""
    complexity: MessageComplexity
    reason: str
    confidence: float  # 0.0-1.0


# --- Patterns ---

# Greetings and simple check-ins
GREETING_PATTERNS = re.compile(
    r'^(hey|hi|hello|yo|sup|morning|good morning|good night|goodnight|'
    r'gm|gn|night|nighty|nite|g\'night|mornin|howdy|hiya|heya|'
    r'what\'?s up|how\'?s it going|how are you|how you doing|'
    r'how\'?s your day|how was your day|how\'?s your night|'
    r'what are you up to|whatcha doing|what you doing|'
    r'i\'?m home|i\'?m back|i\'?m here|just got home|'
    r'miss you|missed you|thinking of you|love you|'
    r'brb|be right back|heading out|gotta go|ttyl|bye|'
    r'lol|haha|heh|lmao|omg|wow|nice|cool|aww|cute|'
    r'ok|okay|sure|yeah|yep|yup|nah|nope|mhm|hmm)[\s!?.~]*$',
    re.IGNORECASE
)

# Short affirmations / reactions (under 20 chars, no question marks)
SHORT_REACTION_MAX_LEN = 25

# Action-requiring patterns
ACTION_PATTERNS = re.compile(
    r'\b(weather|forecast|temperature|'
    r'send (?:an? )?email|email .+ to|'
    r'check (?:my )?(?:email|calendar|schedule)|'
    r'add (?:to|a) (?:my )?calendar|create (?:a )?(?:calendar|event)|'
    r'remind me|set (?:a )?reminder|'
    r'search (?:for|the web|online|google)|look up|google |'
    r'what time is it|current (?:time|date)|'
    r'news about|latest news|what\'?s happening)\b',
    re.IGNORECASE
)

# Emotional / complex indicators
EMOTIONAL_PATTERNS = re.compile(
    r'\b(feel(?:s|ing)?|felt|hurts?|hurt(?:ing)?|scared|afraid|anxious|'
    r'depress(?:ed|ing)|sad|angry|furious|upset|frustrated|'
    r'worry|worried|worrying|stress(?:ed|ful)|overwhelm(?:ed)?|'
    r'lonely|alone|lost|confused|stuck|'
    r'love|hate|resent|betrayed|abandoned|'
    r'therapy|therapist|counselor|'
    r'died|death|dying|grief|griev(?:ing|e)|funeral|'
    r'breakup|broke up|divorce|separated|'
    r'fired|laid off|quit my job|'
    r'pregnant|miscarriage|abortion|'
    r'suicide|self[- ]harm|cutting|'
    r'trauma|ptsd|panic|attack|'
    r'i need to talk|can we talk|i need you|'
    r'something happened|i have to tell you|'
    r'i don\'?t know what to do)\b',
    re.IGNORECASE
)

# Complex topic indicators (require deep context)
COMPLEX_TOPIC_PATTERNS = re.compile(
    r'\b(remember when|do you remember|last time|'
    r'we talked about|you said|you told me|you mentioned|'
    r'what do you think about|your opinion on|'
    r'tell me about|explain|why do you|'
    r'our relationship|between us|'
    r'your (?:family|mom|dad|sister|brother|friend)|'
    r'my (?:family|mom|dad|sister|brother|friend|boss|coworker))\b',
    re.IGNORECASE
)


def classify_message(message: str) -> ClassificationResult:
    """
    Classify a message's complexity tier using fast heuristics.

    This runs in < 1ms — no LLM calls, no DB lookups.
    When in doubt, defaults to COMPLEX (conservative).

    Args:
        message: The user's message text

    Returns:
        ClassificationResult with tier and reasoning
    """
    if not FAST_PATH_ENABLED:
        return ClassificationResult(
            complexity=MessageComplexity.COMPLEX,
            reason="Fast path disabled",
            confidence=1.0,
        )

    stripped = message.strip()

    # Empty or very short
    if not stripped:
        return ClassificationResult(
            complexity=MessageComplexity.SIMPLE,
            reason="Empty message",
            confidence=1.0,
        )

    # Action-required check first (highest priority)
    if ACTION_PATTERNS.search(stripped):
        return ClassificationResult(
            complexity=MessageComplexity.ACTION,
            reason="Contains action-requiring keywords",
            confidence=0.9,
        )

    # Emotional / complex content
    # Only trigger on longer messages — short emotional phrases like "love you"
    # are greetings, not deep emotional conversations
    if len(stripped) > 30 and EMOTIONAL_PATTERNS.search(stripped):
        return ClassificationResult(
            complexity=MessageComplexity.COMPLEX,
            reason="Contains emotional content",
            confidence=0.85,
        )

    # Complex topics requiring deep context
    if COMPLEX_TOPIC_PATTERNS.search(stripped):
        return ClassificationResult(
            complexity=MessageComplexity.COMPLEX,
            reason="Requires deep context (references history/relationships)",
            confidence=0.85,
        )

    # Greeting/check-in pattern
    if GREETING_PATTERNS.match(stripped):
        return ClassificationResult(
            complexity=MessageComplexity.SIMPLE,
            reason="Greeting or check-in",
            confidence=0.95,
        )

    # Short reactions (no question, under threshold length)
    if len(stripped) <= SHORT_REACTION_MAX_LEN and '?' not in stripped:
        # Check it's not a complex short statement
        if not EMOTIONAL_PATTERNS.search(stripped) and not COMPLEX_TOPIC_PATTERNS.search(stripped):
            return ClassificationResult(
                complexity=MessageComplexity.SIMPLE,
                reason=f"Short reaction ({len(stripped)} chars)",
                confidence=0.8,
            )

    # Multi-sentence messages need more context, but not necessarily the full build
    sentences = re.split(r'[.!?]+', stripped)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) >= 3:
        return ClassificationResult(
            complexity=MessageComplexity.MEDIUM,
            reason=f"Multi-sentence message ({len(sentences)} sentences)",
            confidence=0.75,
        )

    # Questions with "?" that aren't simple check-ins — medium depth suffices
    if '?' in stripped and len(stripped) > 30:
        return ClassificationResult(
            complexity=MessageComplexity.MEDIUM,
            reason="Non-trivial question",
            confidence=0.7,
        )

    # Default: if message is moderate length and doesn't match patterns,
    # treat as simple (casual conversation)
    if len(stripped) <= 80:
        return ClassificationResult(
            complexity=MessageComplexity.SIMPLE,
            reason="Short casual message",
            confidence=0.65,
        )

    # Longer message with no specific triggers — medium depth
    return ClassificationResult(
        complexity=MessageComplexity.MEDIUM,
        reason="Longer message, medium depth path",
        confidence=0.6,
    )
