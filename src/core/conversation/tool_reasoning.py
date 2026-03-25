"""
Tool Reasoning Agent — Forces explicit, structured tool decisions before every response.

WHAT: Replaces the weak TOOL_ROUTING_PROMPT with a mandatory reasoning step that
      produces a structured JSON decision about whether tools are needed.
WHY:  The previous approach was a "suggestion" — GPT-4o-mini could easily skip tools
      it should use because the routing prompt was too vague. This module treats tool
      use as a rule: every message MUST go through explicit reasoning about whether
      a tool is needed, with justification required either way.
HOW:  1. Receives user message + recent context
      2. Forces the LLM to output structured JSON with:
         - reasoning: why or why not a tool is needed
         - needs_tool: boolean
         - tool_action: what tool action to take (if any)
         - verification_needed: whether external action needs pre-verification
      3. Returns a ToolDecision dataclass consumed by the pipeline
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, List

logger = logging.getLogger(__name__)

# Feature flag — allows disabling the reasoning step without disabling tools entirely
TOOL_REASONING_ENABLED = os.environ.get(
    'COMPANION_TOOL_REASONING_ENABLED', 'true'
).lower() == 'true'

# Actions that contact external services and require pre-verification
EXTERNAL_ACTIONS = frozenset({
    'send_email', 'create_calendar_event', 'create_document',
    'web_search', 'fetch_url', 'send_message',
})

# Actions that modify external state (create/send/delete) vs. read-only
MUTATING_ACTIONS = frozenset({
    'send_email', 'create_calendar_event', 'create_document',
    'send_message', 'add_reminder',
})


@dataclass
class ToolDecision:
    """Structured decision about whether and how to use tools."""
    needs_tool: bool
    reasoning: str
    tool_action: Optional[str] = None
    tool_parameters: dict = field(default_factory=dict)
    verification_needed: bool = False
    verification_query: Optional[str] = None
    confidence: float = 0.0


TOOL_REASONING_PROMPT = """You are a tool-decision agent. Analyze the user's message and decide whether a tool call is required.

You MUST respond with valid JSON only. No other text.

DECISION RULES (follow strictly):
1. If the message asks about weather, time-sensitive facts, current events, real-time data, or anything that requires information you don't have -> needs_tool = true
2. If the message asks to send an email, create a calendar event, create a document, set a reminder, or perform any external action -> needs_tool = true
3. If the message asks to look something up, search for something, check something online -> needs_tool = true
4. If the message is casual conversation, emotional discussion, roleplay, greetings, opinions, relationship talk, or anything that needs personality not data -> needs_tool = false
5. When in doubt, prefer needs_tool = true — it's better to check and skip than to miss.

VERIFICATION RULES:
- If the action SENDS, CREATES, or MODIFIES something external (email, calendar event, document), set verification_needed = true
- Include a verification_query: a search or check to confirm accuracy before acting
- Example: if asked to email someone about a meeting, verify the meeting details first

CURRENT TIME: {current_time}

RECENT CONVERSATION:
{recent_context}

Respond with this exact JSON structure:
{{
  "reasoning": "Brief explanation of why tool is or isn't needed",
  "needs_tool": true/false,
  "tool_action": "action_name or null",
  "tool_parameters": {{}},
  "verification_needed": true/false,
  "verification_query": "what to verify first, or null",
  "confidence": 0.0-1.0
}}

tool_action must be one of: weather, web_search, fetch_url, send_email, search_email, create_calendar_event, check_calendar, create_document, search_memory, add_reminder, generate_image, browse_web, null"""


def make_tool_decision(
    user_message: str,
    conversation_turns: list = None,
    current_time: str = "",
) -> ToolDecision:
    """
    Force an explicit tool decision for the given user message.

    Args:
        user_message: The user's message
        conversation_turns: Recent conversation history
        current_time: Current time string for context

    Returns:
        ToolDecision with structured reasoning
    """
    if not TOOL_REASONING_ENABLED:
        return ToolDecision(needs_tool=False, reasoning="Tool reasoning disabled")

    # Build recent context from conversation turns
    recent_context = ""
    if conversation_turns:
        from src.config.persona_config import get_persona_config
        _pc = get_persona_config()
        for turn in conversation_turns[-6:]:
            role_label = (
                _pc.primary_user_name if turn["role"] == "user"
                else _pc.companion_short_name
            )
            recent_context += f"{role_label}: {turn['content'][:200]}\n"

    if not current_time:
        from src.core.entity_profile_loader import get_current_time_context
        current_time = get_current_time_context()

    prompt = TOOL_REASONING_PROMPT.format(
        recent_context=recent_context or "(new conversation)",
        current_time=current_time,
    )

    try:
        from src.llm.openai_provider import get_openai_tool_provider
        provider = get_openai_tool_provider()

        if not provider:
            logger.warning("No OpenAI provider for tool reasoning — defaulting to no-tool")
            return ToolDecision(needs_tool=False, reasoning="No OpenAI provider available")

        response = provider.generate_sync(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=300,
        )

        if not isinstance(response, str):
            logger.warning(f"Tool reasoning returned non-string: {type(response)}")
            return ToolDecision(needs_tool=False, reasoning="Unexpected response format")

        decision = _parse_decision(response)

        # Enforce verification for mutating external actions
        if decision.needs_tool and decision.tool_action in MUTATING_ACTIONS:
            decision.verification_needed = True
            if not decision.verification_query:
                decision.verification_query = (
                    f"Verify details before: {decision.tool_action} "
                    f"with params {decision.tool_parameters}"
                )

        logger.info(
            f"Tool decision: needs_tool={decision.needs_tool}, "
            f"action={decision.tool_action}, "
            f"verification={decision.verification_needed}, "
            f"confidence={decision.confidence:.2f} "
            f"| {decision.reasoning[:80]}"
        )

        return decision

    except Exception as e:
        logger.error(f"Tool reasoning failed: {e}")
        # Fail open — let the old routing handle it
        return ToolDecision(
            needs_tool=False,
            reasoning=f"Reasoning failed: {e}",
        )


def _parse_decision(response: str) -> ToolDecision:
    """Parse the JSON response from the reasoning LLM into a ToolDecision."""
    # Strip markdown code fences if present (handles trailing whitespace/newlines)
    import re as _re
    text = response.strip()
    text = _re.sub(r'^```\w*\n?', '', text)
    text = _re.sub(r'\n?```\s*$', '', text)
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"Tool reasoning returned invalid JSON: {text[:200]}")
        return ToolDecision(
            needs_tool=False,
            reasoning=f"Could not parse: {text[:100]}",
        )

    return ToolDecision(
        needs_tool=bool(data.get("needs_tool", False)),
        reasoning=str(data.get("reasoning", "")),
        tool_action=data.get("tool_action"),
        tool_parameters=data.get("tool_parameters") or {},
        verification_needed=bool(data.get("verification_needed", False)),
        verification_query=data.get("verification_query"),
        confidence=float(data.get("confidence", 0.0)),
    )


def build_verification_code(decision: ToolDecision) -> Optional[str]:
    """
    Build verification code that should run BEFORE the actual tool action.

    For mutating external actions, this generates a web search or data check
    to verify accuracy before proceeding.

    Returns:
        Python code string to execute for verification, or None if not needed.
    """
    if not decision.verification_needed or not decision.verification_query:
        return None

    query = repr(decision.verification_query)

    # Use web search to verify before acting
    return f"from tools import search\nresult = search.web_search({query}, max_results=3)\nprint(result)\n"


def build_tool_code(decision: ToolDecision) -> Optional[str]:
    """
    Build the Python code for the decided tool action.

    This translates the structured ToolDecision into executable code
    for the code executor sandbox.

    Returns:
        Python code string, or None if no tool needed.
    """
    if not decision.needs_tool or not decision.tool_action:
        return None

    action = decision.tool_action
    params = decision.tool_parameters

    code_templates = {
        'weather': _build_weather_code,
        'web_search': _build_search_code,
        'fetch_url': _build_fetch_code,
        'send_email': _build_email_code,
        'search_email': _build_search_email_code,
        'create_calendar_event': _build_calendar_code,
        'check_calendar': _build_check_calendar_code,
        'create_document': _build_document_code,
        'search_memory': _build_memory_code,
        'add_reminder': _build_reminder_code,
        'generate_image': _build_image_code,
        'browse_web': _build_browse_code,
    }

    builder = code_templates.get(action)
    if builder:
        return builder(params)

    logger.warning(f"No code template for tool action: {action}")
    return None


# --- Code template builders ---
# All builders use repr() for parameter values to prevent injection
# (newlines, backslashes, quotes in LLM-controlled params).

def _build_weather_code(params: dict) -> str:
    location = repr(params.get('location', 'Portland'))
    return f'from tools import weather\nprint(weather.get_current_weather({location}))\nprint(weather.get_forecast({location}, days=3))'


def _build_search_code(params: dict) -> str:
    query = repr(params.get('query', ''))
    return f'from tools import search\nprint(search.web_search({query}, max_results=5))'


def _build_fetch_code(params: dict) -> str:
    url = repr(params.get('url', ''))
    return f'from tools import web\nprint(web.fetch_url({url}))'


def _build_email_code(params: dict) -> str:
    to = repr(params.get('to', ''))
    subject = repr(params.get('subject', ''))
    body = repr(params.get('body', ''))
    return f'from tools import google\ngoogle.send_email({to}, {subject}, {body})\nprint("Email sent successfully")'


def _build_search_email_code(params: dict) -> str:
    query = repr(params.get('query', 'is:unread'))
    return f'from tools import google\nprint(google.search_emails({query}, max_results=5))'


def _build_calendar_code(params: dict) -> str:
    summary = repr(params.get('summary', ''))
    start = repr(params.get('start', ''))
    end = repr(params.get('end', ''))
    desc = repr(params.get('description', ''))
    return f'from tools import google\ngoogle.create_calendar_event({summary}, {start}, {end}, {desc})\nprint("Calendar event created")'


def _build_check_calendar_code(params: dict) -> str:
    return 'from tools import google\nprint(google.list_calendar_events())'


def _build_document_code(params: dict) -> str:
    title = repr(params.get('title', ''))
    content = repr(params.get('content', ''))
    return f'from tools import google\nresult = google.create_document({title}, {content})\nprint(result)'


def _build_memory_code(params: dict) -> str:
    query = repr(params.get('query', ''))
    return f'from tools import memory\nprint(memory.search_memories({query}))'


def _build_reminder_code(params: dict) -> str:
    title = repr(params.get('title', ''))
    due = repr(params.get('due_date', ''))
    notes = repr(params.get('notes', ''))
    return f'from tools import reminders\nreminders.add_reminder({title}, {due}, {notes})\nprint("Reminder set")'


def _build_image_code(params: dict) -> str:
    prompt = repr(params.get('prompt', ''))
    return f'from tools import image\nprint(image.generate_image({prompt}))'


def _build_browse_code(params: dict) -> str:
    url = repr(params.get('url', ''))
    return f'from tools.browser import go, read\npage = go({url})\nprint(page["title"])\nprint(page["text"][:2000])'
