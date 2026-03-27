"""
Relationship Extractor - Extracts structured relationships from conversation.

WHAT: Scans user messages for explicit relationship statements and converts
them into typed, confidence-scored relationship records. Uses a two-stage
approach: fast keyword pre-filter, then LLM extraction for messages that
pass the filter.

WHY: Graphiti's knowledge graph stores all relationships as generic
RELATES_TO edges, which makes it hard to distinguish "James is Jesse's
father" from "James knows about Jesse's school." This extractor solves the
"trash relationship" problem by requiring explicit RelationshipType enums
(parent_of, married_to, works_at, etc.) and confidence scores.

HOW it fits:
  - The message handler calls extract_relationships_from_message() after
    each conversation turn.
  - The extractor first checks has_relationship_indicators() (fast keyword
    scan). If no keywords match, it skips the LLM call entirely.
  - Extracted relationships are stored via relationship_store.py with
    deduplication (repeat mentions reinforce confidence).

Pipeline: message -> keyword filter -> LLM extraction -> validation
  against RelationshipType enum -> store via RelationshipStore
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from src.config.models import FIREWORKS_DEFAULT_MODEL as FIREWORKS_MODEL

logger = logging.getLogger(__name__)


# Keyword patterns for fast pre-filtering (checked before invoking the LLM).
# If none of these match, the message is skipped entirely.
RELATIONSHIP_PATTERNS = {
    'parent_of': [
        'my son', 'my daughter', 'my kid', 'my child', 'my children',
        'father of', 'mother of', 'parent of', 'dad to', 'mom to',
        "son's name is", "daughter's name is",
    ],
    'child_of': [
        'my father', 'my mother', 'my dad', 'my mom', 'my parent',
        'son of', 'daughter of', 'child of',
    ],
    'married_to': [
        'my wife', 'my husband', 'married to', 'spouse',
        'legally married', 'still married',
    ],
    'separated_from': [
        'separated from', 'estranged from', 'separated but',
        'not living together', 'split from',
    ],
    'divorced_from': [
        'divorced from', 'ex-wife', 'ex-husband', 'former spouse',
    ],
    'partner_of': [
        'my girlfriend', 'my boyfriend', 'my partner',
        'dating', 'together with', 'in a relationship with',
    ],
    'sibling_of': [
        'my sister', 'my brother', 'sibling',
        "sister's name", "brother's name",
    ],
    'works_at': [
        'work at', 'works at', 'employed at', 'job at',
        'work for', 'works for',
    ],
    'coworker_of': [
        'coworker', 'colleague', 'work with', 'works with',
    ],
    'friend_of': [
        'my friend', 'friends with', 'best friend',
    ],
    'owns_pet': [
        'my cat', 'my dog', 'my pet', 'our cat', 'our dog',
    ],
    'lives_with': [
        'live with', 'lives with', 'roommate', 'housemate',
        'staying with', 'moved in with',
    ],
}


def has_relationship_indicators(text: str) -> bool:
    """Quick check if text might contain relationship info."""
    text_lower = text.lower()
    for patterns in RELATIONSHIP_PATTERNS.values():
        for pattern in patterns:
            if pattern in text_lower:
                return True
    return False


def detect_likely_relationship_type(text: str) -> Optional[str]:
    """Detect most likely relationship type from patterns."""
    text_lower = text.lower()
    for rel_type, patterns in RELATIONSHIP_PATTERNS.items():
        for pattern in patterns:
            if pattern in text_lower:
                return rel_type
    return None


class RelationshipExtractor:
    """
    Extracts structured relationships from conversation text.

    Uses LLM for accurate extraction with explicit relationship types.
    """

    EXTRACTION_PROMPT = """/no_think
Extract any EXPLICIT relationships mentioned in this conversation.

CONVERSATION:
{conversation}

RELATIONSHIP TYPES (use ONLY these):
- parent_of: Parent to child (e.g., "my son Jesse")
- child_of: Child to parent (e.g., "my mom Carol")
- married_to: Legal marriage (e.g., "my wife Alia")
- separated_from: Separated but still married
- divorced_from: Legally divorced
- partner_of: Romantic partner, not married (e.g., "my girlfriend {companion_name}")
- sibling_of: Brother/sister relationship
- works_at: Employment (e.g., "I work at Cavallo")
- coworker_of: Work colleague
- friend_of: Friendship
- owns_pet: Pet ownership (e.g., "my cat Tuck")
- lives_with: Living arrangement

RULES:
1. Only extract EXPLICITLY stated relationships
2. Do NOT infer relationships that aren't directly stated
3. Include confidence (0.5-1.0) based on how clear the statement was
4. The speaker in the conversation is {speaker_name} (unless otherwise indicated)

Return JSON array (empty if no relationships found):
[
  {{
    "source": "{speaker_name}",
    "type": "parent_of",
    "target": "Jesse",
    "confidence": 0.95,
    "evidence": "my son Jesse"
  }}
]

Extract relationships now (JSON only):"""

    def __init__(self):
        self._client = None

    def _get_client(self):
        """Get Fireworks LLM client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url="https://api.fireworks.ai/inference/v1",
                api_key=os.environ.get('FIREWORKS_API_KEY')
            )
        return self._client

    def extract_relationships(
        self,
        user_message: str,
        companion_response: str = None,
        speaker_name: str = None
    ) -> List[Dict[str, Any]]:
        """
        Extract relationships from a conversation exchange.

        Args:
            user_message: Message from user
            companion_response: Optional response from companion
            speaker_name: Name of the speaker (default: from persona config)

        Returns:
            List of relationship dicts with: source, type, target, confidence, evidence
        """
        if speaker_name is None:
            from src.config.persona_config import get_persona_config
            speaker_name = get_persona_config().primary_user_name

        # Quick check - skip if no relationship indicators
        combined_text = user_message
        if companion_response:
            combined_text += "\n" + companion_response

        if not has_relationship_indicators(combined_text):
            return []

        # Format conversation for LLM
        from src.config.persona_config import get_persona_config as _gpc
        companion_name = _gpc().companion_short_name
        conversation = f"{speaker_name}: {user_message}"
        if companion_response:
            conversation += f"\n\n{companion_name}: {companion_response}"

        try:
            client = self._get_client()

            response = client.chat.completions.create(
                model=FIREWORKS_MODEL,
                max_tokens=500,
                temperature=0.1,  # Low temp for accurate extraction
                messages=[{
                    "role": "user",
                    "content": self.EXTRACTION_PROMPT.format(conversation=conversation, speaker_name=speaker_name)
                }]
            )

            content = response.choices[0].message.content.strip()

            # Parse JSON response
            relationships = self._parse_response(content)

            # Validate relationship types
            from src.memory.relationship_store import RelationshipType
            valid_types = {t.value for t in RelationshipType}

            validated = []
            for rel in relationships:
                if rel.get('type') in valid_types:
                    validated.append(rel)
                else:
                    logger.warning(f"Invalid relationship type: {rel.get('type')}")

            logger.info(f"Extracted {len(validated)} relationships from message")
            return validated

        except Exception as e:
            logger.error(f"Relationship extraction failed: {e}")
            return []

    def _parse_response(self, content: str) -> List[Dict[str, Any]]:
        """Parse LLM response into relationship list."""
        # Handle thinking tags
        import re
        if '<think>' in content:
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            content = content.strip()

        # Try direct JSON parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try extracting from code blocks
        json_pattern = r'```(?:json)?\s*(\[[\s\S]*?\])\s*```'
        match = re.search(json_pattern, content)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding array brackets
        start = content.find('[')
        end = content.rfind(']') + 1
        if start >= 0 and end > start:
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass

        logger.warning(f"Could not parse relationship extraction response: {content[:100]}")
        return []

    def extract_and_store(
        self,
        user_message: str,
        companion_response: str = None,
        speaker_name: str = None,
        user_email: str = None,
        message_id: int = None
    ) -> int:
        """
        Extract relationships and store them.

        Returns count of relationships stored.
        """
        relationships = self.extract_relationships(user_message, companion_response, speaker_name)

        if not relationships:
            return 0

        from src.memory.relationship_store import get_relationship_store, RelationshipType

        store = get_relationship_store()
        stored = 0

        for rel in relationships:
            try:
                rel_type = RelationshipType(rel['type'])
                rel_id = store.store_relationship(
                    source_entity=rel['source'],
                    relationship_type=rel_type,
                    target_entity=rel['target'],
                    confidence=rel.get('confidence', 0.7),
                    context=rel.get('evidence'),
                    source_message_id=message_id,
                    user_email=user_email
                )

                if rel_id:
                    stored += 1

            except Exception as e:
                logger.error(f"Failed to store relationship {rel}: {e}")

        return stored


# Singleton instance
_extractor: Optional[RelationshipExtractor] = None


def get_relationship_extractor() -> RelationshipExtractor:
    """Get singleton extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = RelationshipExtractor()
    return _extractor


def extract_relationships_from_message(
    user_message: str,
    companion_response: str = None,
    user_email: str = None,
    message_id: int = None
) -> int:
    """
    Main entry point - extract and store relationships from a message.

    Called from message handler after each conversation turn.
    Returns count of relationships stored.
    """
    from src.config.persona_config import get_persona_config
    extractor = get_relationship_extractor()
    return extractor.extract_and_store(
        user_message=user_message,
        companion_response=companion_response,
        speaker_name=get_persona_config().primary_user_name,
        user_email=user_email,
        message_id=message_id
    )
