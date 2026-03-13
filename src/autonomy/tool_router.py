"""
Tool Router -- Routes goal-step actions to concrete tool invocations.

WHAT: Maps each GoalStep action_type to the appropriate external service or
      internal handler:
        research      -> Tavily web search (with LLM synthesis)
        prepare       -> LLM content generation (internal)
        create        -> Google Docs API
        notify        -> Gmail API
        calendar_check -> n8n webhook (or Google Calendar)
        calendar_add  -> Google Calendar API
        being         -> Background-life activity recording (no-op externally)
        suppress      -> Pure no-op (reduces action budget)
        relate        -> No-op here (handled during live conversation)

WHY:  Goal steps are abstract ("research X").  The router turns them into
      concrete API calls and returns a ToolResult the outcome tracker can record.

HOW IT FITS:
  - Called by the autonomous action task after ActionBudget approves a step.
  - ToolResult is passed to OutcomeTracker.record_action_outcome().
  - Falls back to LLM knowledge when Tavily API key is missing.
  - Google integrations use the companion's own credentials via google_service.
"""

import os
import json
import logging
import requests
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

PST = ZoneInfo('America/Los_Angeles')


# =============================================================================
# Data model
# =============================================================================

@dataclass
class ToolResult:
    """Result from a tool invocation."""
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    source: str = ''  # which tool was used (e.g., 'tavily', 'google:docs')


# =============================================================================
# ToolRouter
# =============================================================================

class ToolRouter:
    """Routes goal step actions to the appropriate tool."""

    def __init__(self):
        self.n8n_base_url = os.environ.get('N8N_HOST', 'http://companion-n8n:5678')
        self.n8n_api_key = os.environ.get('N8N_API_KEY', '')

    async def execute(self, step, context: Dict = None) -> ToolResult:
        """Execute a goal step using the appropriate tool.

        Args:
            step: GoalStep with action_type and parameters
            context: Additional context (user_email, etc.)

        Returns:
            ToolResult with success/failure and data
        """
        context = context or {}
        action_type = step.action_type
        params = step.parameters or {}

        routes = {
            'research': self._do_research,
            'prepare': self._do_prepare,
            'create': self._do_create,
            'notify': self._do_notify,
            'calendar_check': self._do_cal_check,
            'calendar_add': self._do_cal_add,
            'being': self._do_being,
            'suppress': self._do_suppress,
            'relate': self._noop,
        }

        handler = routes.get(action_type)
        if not handler:
            return ToolResult(
                success=False,
                data={},
                error=f"Unknown action type: {action_type}",
                source='unknown'
            )

        try:
            return await handler(params, context)
        except Exception as e:
            logger.error(f"Tool execution failed for {action_type}: {e}")
            return ToolResult(
                success=False,
                data={},
                error=str(e),
                source=action_type
            )

    def execute_sync(self, step, context: Dict = None) -> ToolResult:
        """Synchronous wrapper for execute()."""
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.execute(step, context))
        finally:
            loop.close()

    # =========================================================================
    # Research (Tavily web search)
    # =========================================================================

    async def _do_research(self, params: Dict, context: Dict) -> ToolResult:
        """Research a topic via Tavily web search."""
        topic = params.get('topic', '')
        if not topic:
            return ToolResult(success=False, data={}, error="No topic specified", source='research')

        api_key = os.environ.get('TAVILY_API_KEY')
        if not api_key:
            # Fall back to LLM knowledge
            return await self._do_prepare(
                {'topic': topic, 'instruction': f'Research and summarize: {topic}'},
                context
            )

        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": topic,
                    "search_depth": "basic",
                    "include_answer": True,
                    "max_results": 3
                },
                timeout=15
            )

            if response.status_code == 200:
                result = response.json()
                answer = result.get('answer', '')
                sources = [
                    {"title": r.get('title', ''), "snippet": r.get('content', '')[:200]}
                    for r in result.get('results', [])[:3]
                ]

                # Synthesize into a natural finding
                from src.llm.provider_factory import generate_sync
                synthesis = generate_sync(
                    messages=[{"role": "user", "content": f"""The companion researched: {topic}

Search results:
{answer}

{json.dumps(sources, indent=2)}

Synthesize into a brief, interesting insight (under 100 words).
Something she might say "I was looking into X and found out..."
Be specific based on the actual results."""}],
                    temperature=0.7,
                    max_tokens=150
                )

                return ToolResult(
                    success=True,
                    data={
                        'topic': topic,
                        'finding': synthesis.strip() if synthesis else answer,
                        'raw_answer': answer,
                        'sources': sources
                    },
                    source='tavily'
                )

            return ToolResult(success=False, data={}, error=f"Tavily returned {response.status_code}", source='tavily')

        except Exception as e:
            logger.warning(f"Tavily search failed: {e}")
            return ToolResult(success=False, data={}, error=str(e), source='tavily')

    # =========================================================================
    # Prepare (LLM content generation)
    # =========================================================================

    async def _do_prepare(self, params: Dict, context: Dict) -> ToolResult:
        """Generate content using LLM."""
        topic = params.get('topic', '')
        instruction = params.get('instruction', f'Prepare thoughts about: {topic}')

        try:
            from src.llm.provider_factory import generate_sync

            response = generate_sync(
                messages=[{"role": "user", "content": instruction}],
                temperature=0.7,
                max_tokens=300
            )

            if response:
                return ToolResult(
                    success=True,
                    data={'content': response.strip(), 'topic': topic},
                    source='llm'
                )

            return ToolResult(success=False, data={}, error="No LLM response", source='llm')

        except Exception as e:
            return ToolResult(success=False, data={}, error=str(e), source='llm')

    # =========================================================================
    # n8n Workflows (Gmail, Calendar, Docs)
    # =========================================================================

    def _invoke_n8n(self, workflow_type: str, payload: Dict) -> Dict:
        """Call n8n webhook to trigger a workflow."""
        webhook_url = f"{self.n8n_base_url}/webhook/{workflow_type}"
        headers = {"Content-Type": "application/json"}

        if self.n8n_api_key:
            headers["Authorization"] = f"Bearer {self.n8n_api_key}"

        try:
            response = requests.post(webhook_url, json=payload, headers=headers, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"n8n webhook {workflow_type} returned {response.status_code}: {response.text[:200]}")
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            logger.warning(f"n8n webhook {workflow_type} failed: {e}")
            return {"error": str(e)}

    async def _do_create(self, params: Dict, context: Dict) -> ToolResult:
        """Create content via Google Docs API."""
        title = params.get('title', 'Companion Note')
        content = params.get('content', '')

        if not content:
            # Generate content first
            topic = params.get('topic', title)
            prep = await self._do_prepare({'topic': topic}, context)
            if prep.success:
                content = prep.data.get('content', '')

        try:
            from src.integrations.google_service import get_google_service
            result = get_google_service().create_document(title=title, content=content)

            # Store doc reference for future awareness
            try:
                doc_ref = {
                    'documentId': result.get('documentId'),
                    'title': title,
                    'url': result.get('url'),
                    'created_at': datetime.now(PST).isoformat(),
                    'topic': params.get('topic', title),
                }
                docs_file = os.path.join(
                    os.environ.get('DATA_DIR', '/app/data'), 'companion_docs.json'
                )
                docs = []
                if os.path.exists(docs_file):
                    with open(docs_file, 'r') as f:
                        docs = json.load(f)
                docs.append(doc_ref)
                docs = docs[-20:]  # Keep last 20
                with open(docs_file, 'w') as f:
                    json.dump(docs, f, indent=2)
            except Exception:
                pass

            return ToolResult(
                success=True,
                data={'doc_url': result.get('url', ''), 'title': title, 'documentId': result.get('documentId')},
                source='google:docs'
            )
        except Exception as e:
            logger.error(f"Google Docs creation failed: {e}")
            return ToolResult(success=False, data={}, error=str(e), source='google:docs')

    async def _do_notify(self, params: Dict, context: Dict) -> ToolResult:
        """Send notification via Gmail API."""
        to = params.get('to', '')
        subject = params.get('subject', '')
        body = params.get('body', '')

        if not all([to, subject, body]):
            return ToolResult(success=False, data={}, error="Missing to/subject/body", source='google:gmail')

        try:
            from src.integrations.google_service import get_google_service
            result = get_google_service().send_email(to=to, subject=subject, body=body)
            return ToolResult(
                success=True,
                data={'sent_to': to, 'subject': subject, 'messageId': result.get('id')},
                source='google:gmail'
            )
        except Exception as e:
            logger.error(f"Gmail send failed: {e}")
            return ToolResult(success=False, data={}, error=str(e), source='google:gmail')

    async def _do_cal_check(self, params: Dict, context: Dict) -> ToolResult:
        """Check calendar via n8n."""
        result = self._invoke_n8n('calendar-list', {
            'time_min': params.get('time_min', datetime.now(PST).isoformat()),
            'time_max': params.get('time_max', ''),
            'max_results': params.get('max_results', 10)
        })

        if 'error' not in result:
            return ToolResult(
                success=True,
                data={'events': result.get('events', result)},
                source='n8n:calendar'
            )
        return ToolResult(success=False, data={}, error=result.get('error'), source='n8n:calendar')

    async def _do_cal_add(self, params: Dict, context: Dict) -> ToolResult:
        """Add calendar event via Google Calendar API.

        Uses the companion's own calendar (from data/companion_calendar_id.txt) if available,
        otherwise falls back to 'primary'.
        """
        summary = params.get('summary', '')
        start = params.get('start', '')
        end = params.get('end', '')
        description = params.get('description', '')
        location = params.get('location', '')

        if not all([summary, start, end]):
            return ToolResult(success=False, data={}, error="Missing summary/start/end", source='google:calendar')

        # Read the companion's calendar ID if available
        calendar_id = 'primary'
        try:
            cal_id_file = os.path.join(
                os.environ.get('DATA_DIR', '/app/data'), 'companion_calendar_id.txt'
            )
            if os.path.exists(cal_id_file):
                with open(cal_id_file, 'r') as f:
                    cal_id = f.read().strip()
                    if cal_id:
                        calendar_id = cal_id
        except Exception:
            pass

        try:
            from src.integrations.google_service import get_google_service
            result = get_google_service().create_calendar_event(
                summary=summary, start=start, end=end,
                description=description, location=location,
                calendar_id=calendar_id,
            )
            return ToolResult(success=True, data={'event': result}, source='google:calendar')
        except Exception as e:
            logger.error(f"Calendar event creation failed: {e}")
            return ToolResult(success=False, data={}, error=str(e), source='google:calendar')

    # =========================================================================
    # Being / Suppress (internal state management)
    # =========================================================================

    async def _do_being(self, params: Dict, context: Dict) -> ToolResult:
        """Record a being/downtime activity. No external action needed."""
        description = params.get('description', 'taking it easy')

        try:
            from src.core.background_life import get_background_life
            bg = get_background_life()
            bg.record_activity(
                activity_type='personal',
                description=description,
                duration_hours=0.5,
                mood='relaxed',
                energy=0.5
            )
        except Exception:
            pass

        return ToolResult(
            success=True,
            data={'activity': description, 'type': 'being'},
            source='internal'
        )

    async def _do_suppress(self, params: Dict, context: Dict) -> ToolResult:
        """Suppress action — this is a no-op that reduces the action budget."""
        return ToolResult(
            success=True,
            data={'suppressed': True, 'reason': params.get('reason', 'being goal active')},
            source='internal'
        )

    async def _noop(self, params: Dict, context: Dict) -> ToolResult:
        """No-op for relate steps (handled in conversation, not here)."""
        return ToolResult(
            success=True,
            data={'note': 'relate steps are handled during conversation'},
            source='noop'
        )


# =============================================================================
# Singleton accessor
# =============================================================================

_router: Optional[ToolRouter] = None


def get_tool_router() -> ToolRouter:
    global _router
    if _router is None:
        _router = ToolRouter()
    return _router
