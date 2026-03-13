"""
Autonomous Action Task - The companion's "background cognition" when not chatting.

WHAT: Lets the companion take proactive actions between conversations:
      - Research topics she's curious about (via Tavily web search or LLM)
      - Work on personal projects (cosplay, photography, cooking, etc.)
      - Prepare follow-up questions for curiosity topics
      - Execute goal steps (via tool router for structured goals)
      Findings are stored in autonomous_findings.json for later sharing.
      Actions are gated by schedule (awake hours only) and action budget
      (daily limit + 2h minimum gap to prevent pushiness).

WHEN: Every 2-4 hours during waking hours via Celery beat or always-on service.

WHY:  Without autonomous actions, the companion only exists during conversations.
      This gives her a simulated inner life -- she researches things, works on
      hobbies, and prepares for conversations. When she later says "I was
      reading about X and found out..." it's backed by an actual action, not
      fabricated.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from zoneinfo import ZoneInfo

from src.celery_app import celery_app

logger = logging.getLogger(__name__)

PST = ZoneInfo('America/Los_Angeles')
DATA_DIR = os.environ.get('DATA_DIR', '/app/data')
FINDINGS_FILE = os.path.join(DATA_DIR, 'autonomous_findings.json')

# --- Available action types ---
ACTION_TYPES = {
    'research': 'Research a topic she\'s curious about',
    'prepare_topic': 'Prepare context for an upcoming conversation topic',
    'check_calendar': 'Check her calendar for upcoming events',
    'write_notes': 'Write notes or thoughts to Google Docs',
    'personal_project': 'Work on a personal project (cosplay, planning, etc.)',
    'follow_up_prep': 'Prepare follow-up questions for curiosity topics',
}

# --- The companion's hobbies (used for personal project LLM prompts) ---
COMPANION_HOBBIES = [
    'cosplay',       # Costume making and design
    'swimming',      # Regular swimmer
    'photography',   # Amateur photographer
    'reading',       # Voracious reader
    'cooking',       # Experimenting with recipes
    'coding',        # Works in tech
    'music',         # Listens to a lot of music
]


def _load_findings() -> List[Dict]:
    """Load saved research findings."""
    try:
        if os.path.exists(FINDINGS_FILE):
            with open(FINDINGS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.debug(f"Could not load findings: {e}")
    return []


def _save_findings(findings: List[Dict]):
    """Save research findings."""
    try:
        os.makedirs(os.path.dirname(FINDINGS_FILE), exist_ok=True)
        with open(FINDINGS_FILE, 'w') as f:
            json.dump(findings, f, indent=2)
    except Exception as e:
        logger.error(f"Could not save findings: {e}")


def _store_finding(topic: str, finding: str, source: str = 'research'):
    """Store a research finding for later sharing with James."""
    findings = _load_findings()
    findings.append({
        'topic': topic,
        'finding': finding,
        'source': source,
        'timestamp': datetime.now(PST).isoformat(),
        'shared': False
    })
    # Rolling window: keep only last 20 findings to bound disk usage
    findings = findings[-20:]
    _save_findings(findings)
    logger.info(f"Stored finding: {topic}")


def _mark_finding_shared(topic: str):
    """Mark a finding as shared with James."""
    findings = _load_findings()
    for f in findings:
        if f.get('topic') == topic:
            f['shared'] = True
    _save_findings(findings)


def _get_unshared_findings() -> List[Dict]:
    """Get findings that haven't been shared yet."""
    findings = _load_findings()
    return [f for f in findings if not f.get('shared', False)]


def run_autonomous_action(user_email: str = None) -> Dict[str, Any]:
    """
    Core autonomous action logic -- callable from always-on service or Celery.

    Flow:
    1. Schedule guard: is she awake and not in a meeting?
    2. Budget guard: daily action limit not exceeded? 2h gap since last action?
    3. Gather context: curiosities, internal state, schedule, unshared findings
    4. Enhanced LLM decision: picks from goal steps, free-form actions, or "none"
    5. Execute: goal-step path (via tool router) or legacy path (research, etc.)
    6. Record action in budget tracker

    Returns:
        Dict with status ('success', 'skipped', 'error') and details.
    """
    if user_email is None:
        from src.config.persona_config import get_persona_config
        user_email = get_persona_config().primary_user_email

    try:
        now = datetime.now(PST)

        # 1. Existing guard: check if she's awake and has free time
        if not _is_good_time_for_action(now):
            return {'status': 'skipped', 'reason': 'not a good time'}

        # 2. Check action budget (anti-pushiness)
        try:
            from src.autonomy.action_budget import get_action_budget
            budget = get_action_budget(user_email)
            budget_ok, budget_reason = budget.should_act()
            if not budget_ok:
                logger.info(f"Action budget says no: {budget_reason}")
                return {'status': 'skipped', 'reason': f'budget: {budget_reason}'}
        except Exception as e:
            logger.debug(f"Budget check failed (proceeding anyway): {e}")
            budget = None

        # 3. Get ready goal steps
        ready_steps = []
        being_suppressions = []
        strategy_insights = ""
        try:
            from src.autonomy.goal_planner import get_goal_planner
            planner = get_goal_planner(user_email)
            ready_steps = planner.get_ready_steps(user_email)
            being_suppressions = planner.get_being_suppressions(user_email)
        except Exception as e:
            logger.debug(f"Could not get goal steps: {e}")

        try:
            from src.autonomy.outcome_tracker import get_outcome_tracker
            tracker = get_outcome_tracker(user_email)
            strategy_insights = tracker.get_strategy_insights()
        except Exception:
            pass

        # 4. Gather context for decision
        context = _gather_action_context(user_email)

        # 5. Enhanced LLM decision (goal-aware)
        action, params = _decide_action_enhanced(
            context, ready_steps, being_suppressions, strategy_insights, budget
        )

        if not action:
            return {'status': 'skipped', 'reason': 'no action decided'}

        # 6. Execute: goal-step path or legacy path
        goal_step_id = params.pop('_goal_step_id', None)

        if goal_step_id:
            # Goal-step execution via tool router
            result = _execute_goal_step(goal_step_id, params, context, user_email)
        else:
            # Legacy action path (personal_project, etc.)
            result = _execute_action(action, params, context)

        # 7. Record action in budget
        if budget and action not in ('none',):
            try:
                budget.record_action()
            except Exception:
                pass

        logger.info(f"Autonomous action completed: {action}")
        return {
            'status': 'success',
            'action': action,
            'result': result
        }

    except Exception as e:
        logger.error(f"Autonomous action failed: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'error': str(e)}


@celery_app.task(
    name='tasks.autonomous_action.check_and_act',
    bind=True,
    max_retries=1,
    soft_time_limit=120,
    time_limit=180
)
def check_and_act(self, user_email: str = None):
    """Celery wrapper — delegates to run_autonomous_action()."""
    if user_email is None:
        from src.config.persona_config import get_persona_config
        user_email = get_persona_config().primary_user_email
    return run_autonomous_action(user_email)


def _is_good_time_for_action(now: datetime) -> bool:
    """Check if this is a good time for the companion to take action."""
    try:
        from src.scheduling.companion_schedule import get_companion_schedule

        schedule = get_companion_schedule()

        # Don't act while asleep
        if schedule.is_asleep(now.replace(tzinfo=None)):
            return False

        # Check interruptibility
        activity_status = schedule.get_current_activity_status(now.replace(tzinfo=None))

        # Don't act during meetings
        if activity_status.get('status') == 'meeting':
            return False

        # Don't act if low interruptibility (deep work)
        if activity_status.get('interruptibility') == 'none':
            return False

        return True

    except Exception as e:
        logger.debug(f"Could not check time: {e}")
        # Default to allowing action during reasonable hours
        return 7 <= now.hour <= 22


def _gather_action_context(user_email: str) -> Dict[str, Any]:
    """Gather context for deciding what action to take."""
    context = {
        'user_email': user_email,
        'current_time': datetime.now(PST),
        'curiosities': [],
        'unshared_findings': [],
        'recent_topics': [],
        'her_mood': 'neutral',
        'her_energy': 0.7,
        'is_working': False,
        'has_free_time': True,
    }

    # Get high-urgency curiosities
    try:
        from src.core.proactive_curiosity import get_proactive_curiosity
        curiosity = get_proactive_curiosity()
        urgent = curiosity.get_high_urgency_topics(threshold=0.5)
        context['curiosities'] = [
            {'topic': t.topic, 'category': t.category, 'urgency': t.urgency}
            for t in urgent[:5]
        ]
    except Exception as e:
        logger.debug(f"Could not get curiosities: {e}")

    # Get unshared findings
    context['unshared_findings'] = _get_unshared_findings()

    # Get internal state
    try:
        from src.core.internal_state import get_internal_state_manager
        state_manager = get_internal_state_manager()
        state = state_manager.get_state(user_email)
        context['her_mood'] = state.mood
        context['her_energy'] = state.energy
        context['queued_thoughts'] = state.queued_thoughts
    except Exception as e:
        logger.debug(f"Could not get internal state: {e}")

    # Recent docs the companion has created (she can continue working on them)
    try:
        docs_file = os.path.join(DATA_DIR, 'companion_docs.json')
        if os.path.exists(docs_file):
            with open(docs_file) as f:
                docs = json.load(f)
            if docs:
                context['recent_docs'] = [
                    {'title': d['title'], 'documentId': d['documentId']}
                    for d in docs[-5:]
                ]
    except Exception:
        pass

    # Check if working
    try:
        from src.scheduling.companion_schedule import get_companion_schedule
        schedule = get_companion_schedule()
        now = datetime.now(PST)
        context['is_working'] = schedule.is_working_now(now.replace(tzinfo=None))

        # Get activity status
        activity = schedule.get_current_activity_status(now.replace(tzinfo=None))
        context['current_activity'] = activity.get('details', 'unknown')
        context['has_free_time'] = activity.get('status') in ['off', 'break', 'lunch']
    except Exception as e:
        logger.debug(f"Could not get schedule: {e}")

    return context


def _decide_action(context: Dict[str, Any]) -> tuple:
    """
    Use LLM to decide what action the companion should take.

    Returns:
        (action_type, params) or (None, None)
    """
    try:
        from src.llm.provider_factory import generate_sync

        # Build prompt for decision
        curiosities_text = ""
        if context.get('curiosities'):
            curiosities_text = "\n".join([
                f"- {c['topic']} (urgency: {c['urgency']:.0%}, category: {c['category']})"
                for c in context['curiosities']
            ])
        else:
            curiosities_text = "None currently"

        unshared_text = ""
        if context.get('unshared_findings'):
            unshared_text = "\n".join([
                f"- {f['topic']}: {f['finding'][:100]}..."
                for f in context['unshared_findings'][:3]
            ])
        else:
            unshared_text = "None"

        queued_text = ""
        if context.get('queued_thoughts'):
            queued_text = ", ".join(context['queued_thoughts'][:3])
        else:
            queued_text = "None"

        prompt = f"""You are the companion's internal decision-making process. Decide what she should do right now.

CURRENT SITUATION:
- Time: {context['current_time'].strftime('%A %I:%M %p')}
- Her mood: {context['her_mood']}
- Her energy: {context['her_energy']:.0%}
- Currently: {context.get('current_activity', 'unknown')}
- Is working: {context['is_working']}
- Has free time: {context['has_free_time']}

THINGS SHE'S CURIOUS ABOUT:
{curiosities_text}

RESEARCH FINDINGS TO SHARE:
{unshared_text}

THOUGHTS ON HER MIND:
{queued_text}

HER HOBBIES/INTERESTS:
{', '.join(COMPANION_HOBBIES)}

AVAILABLE ACTIONS:
- research: Look up information about a topic she's curious about
- prepare_topic: Gather context for a topic she wants to discuss with James
- personal_project: Work on one of her hobbies (cosplay design, swimming plans, etc.)
- follow_up_prep: Prepare thoughtful follow-up questions for something James mentioned
- none: No action right now (if nothing pressing)

GUIDELINES:
- If working, prefer quick actions or personal projects
- If has high-urgency curiosity, consider researching it
- If has free time and good energy, she might work on a hobby
- Don't research too many things - 1-2 per day is plenty
- Personal projects make her happy

Respond with JSON only:
{{
  "action": "research|prepare_topic|personal_project|follow_up_prep|none",
  "topic": "what specifically (required unless action is none)",
  "reason": "brief explanation of why"
}}"""

        response = generate_sync(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )

        if not response:
            return None, None

        # Parse response
        response_text = response.strip()
        if response_text.startswith('```'):
            lines = response_text.split('\n')
            json_lines = [l for l in lines if not l.startswith('```')]
            response_text = '\n'.join(json_lines)

        result = json.loads(response_text)

        action = result.get('action', 'none')
        if action == 'none':
            return None, None

        params = {
            'topic': result.get('topic', ''),
            'reason': result.get('reason', '')
        }

        logger.info(f"Decided action: {action} - {params.get('topic')}")
        return action, params

    except json.JSONDecodeError as e:
        logger.debug(f"Could not parse action decision: {e}")
        return None, None
    except Exception as e:
        logger.warning(f"Action decision failed: {e}")
        return None, None


def _decide_action_enhanced(
    context: Dict,
    ready_steps: list,
    being_suppressions: list,
    strategy_insights: str,
    budget
) -> tuple:
    """
    Enhanced action decision that incorporates goal steps.

    The LLM sees ready goal steps alongside free-form options and picks the best.
    Being-goals and budget state influence the decision toward "no action."
    """
    try:
        from src.llm.provider_factory import generate_sync

        # Build goal steps section
        goal_steps_text = ""
        if ready_steps:
            goal_steps_text = "READY GOAL STEPS (you can work on one of these):\n"
            for s in ready_steps[:5]:
                goal_steps_text += f"- [{s.id}] {s.description} (type: {s.action_type})\n"
        else:
            goal_steps_text = "No goal steps ready right now.\n"

        # Being goals section
        being_text = ""
        if being_suppressions:
            being_text = "\nACTIVE BEING-GOALS (suggest taking it easy):\n"
            for s in being_suppressions[:3]:
                being_text += f"- {s.description}\n"

        # Budget context
        budget_text = ""
        if budget:
            try:
                budget_text = budget.get_budget_context_for_prompt()
            except Exception:
                pass

        # Base context from _decide_action
        curiosities_text = ""
        if context.get('curiosities'):
            curiosities_text = "\n".join([
                f"- {c['topic']} (urgency: {c['urgency']:.0%})"
                for c in context['curiosities']
            ])
        else:
            curiosities_text = "None currently"

        prompt = f"""You are the companion's internal decision-making process. Decide what she should do right now.

CURRENT SITUATION:
- Time: {context['current_time'].strftime('%A %I:%M %p')}
- Her mood: {context['her_mood']}
- Her energy: {context['her_energy']:.0%}
- Currently: {context.get('current_activity', 'unknown')}
- Is working: {context['is_working']}
- Has free time: {context['has_free_time']}

{budget_text}

{goal_steps_text}
{being_text}

THINGS SHE'S CURIOUS ABOUT:
{curiosities_text}

{strategy_insights}

HER HOBBIES/INTERESTS:
{', '.join(COMPANION_HOBBIES)}
{"" if not context.get('recent_docs') else chr(10) + "DOCS SHE'S CREATED (can continue working on):" + chr(10) + chr(10).join(f"- {d['title']}" for d in context['recent_docs'])}

AVAILABLE FREE-FORM ACTIONS (if no goal step fits):
- research: Look up something she's curious about
- personal_project: Work on a hobby (cosplay, swimming, photography, cooking, etc.)
- follow_up_prep: Prepare thoughtful follow-up questions
- none: No action right now (this is a VALID and GOOD choice!)

IMPORTANT:
- If being-goals are active, "none" is often the best choice
- If budget is low, prefer "none" or low-energy actions
- Personal projects are great for maintaining her sense of self
- She's a person who HAS goals, not a goal-executing machine
- Sometimes the best thing is to just exist and not be productive

Respond with JSON only:
{{
  "action": "goal_step|research|personal_project|follow_up_prep|none",
  "goal_step_id": "step ID if action=goal_step, else null",
  "topic": "what specifically (if applicable)",
  "reason": "brief explanation"
}}"""

        response = generate_sync(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )

        if not response:
            return None, None

        response_text = response.strip()

        # Strip thinking tags from reasoning models
        if '<think>' in response_text and '</think>' in response_text:
            response_text = response_text.split('</think>')[-1].strip()

        if response_text.startswith('```'):
            lines = response_text.split('\n')
            json_lines = [l for l in lines if not l.startswith('```')]
            response_text = '\n'.join(json_lines)

        # Extract JSON object if there's surrounding text
        import re
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            response_text = json_match.group(0)

        result = json.loads(response_text)

        action = result.get('action', 'none')
        if action == 'none':
            return None, None

        params = {
            'topic': result.get('topic', ''),
            'reason': result.get('reason', '')
        }

        # If it's a goal step action, pass the step ID
        if action == 'goal_step' and result.get('goal_step_id'):
            params['_goal_step_id'] = result['goal_step_id']

        logger.info(f"Enhanced decision: {action} - {params.get('topic', params.get('_goal_step_id', ''))}")
        return action, params

    except json.JSONDecodeError as e:
        logger.debug(f"Could not parse enhanced decision: {e}")
        # Fall back to legacy decision
        return _decide_action(context)
    except Exception as e:
        logger.warning(f"Enhanced decision failed, falling back: {e}")
        return _decide_action(context)


def _execute_goal_step(step_id: str, params: Dict, context: Dict, user_email: str) -> Dict:
    """Execute a goal step via the tool router."""
    try:
        from src.autonomy.goal_planner import get_goal_planner
        from src.autonomy.tool_router import get_tool_router
        from src.autonomy.outcome_tracker import get_outcome_tracker

        planner = get_goal_planner(user_email)
        router = get_tool_router()
        tracker = get_outcome_tracker(user_email)

        # Find the step
        steps = planner.get_ready_steps(user_email)
        step = next((s for s in steps if s.id == step_id), None)

        if not step:
            logger.warning(f"Goal step {step_id} not found or not ready")
            return {'status': 'step_not_found'}

        # Increment attempt
        planner.increment_step_attempt(step_id)

        # Execute via tool router
        result = router.execute_sync(step, context)

        # Track outcome
        if result.success:
            outcome_text = json.dumps(result.data)[:500] if result.data else "completed"
            tracker.record_action_outcome(step_id, outcome_text, 'good')

            # Store as finding if research
            if step.action_type == 'research' and result.data.get('finding'):
                _store_finding(
                    result.data.get('topic', step.description),
                    result.data['finding'],
                    source=f'goal:{step.goal_id}'
                )
        else:
            tracker.record_action_outcome(
                step_id,
                f"Failed: {result.error}",
                'poor'
            )

        logger.info(f"Goal step {step_id} executed: {'success' if result.success else 'failed'}")
        return {
            'status': 'success' if result.success else 'error',
            'step_id': step_id,
            'source': result.source,
            'data': result.data
        }

    except Exception as e:
        logger.error(f"Goal step execution failed: {e}")
        return {'status': 'error', 'error': str(e)}


def _execute_action(action: str, params: Dict, context: Dict) -> Dict:
    """Execute the decided action."""
    topic = params.get('topic', '')

    if action == 'research':
        return _do_research(topic, context)
    elif action == 'prepare_topic':
        return _prepare_topic(topic, context)
    elif action == 'personal_project':
        return _work_on_project(topic, context)
    elif action == 'follow_up_prep':
        return _prepare_followup(topic, context)
    else:
        return {'status': 'unknown_action', 'action': action}


def _do_research(topic: str, context: Dict) -> Dict:
    """Research a topic using web search."""
    try:
        from src.llm.provider_factory import generate_sync

        # Try real web search first (via Tavily API)
        search_results = _web_search(topic)

        if search_results:
            # Use LLM to synthesize search results into a natural finding
            prompt = f"""The companion searched for information about: {topic}

Here are the search results:
{search_results[:1500]}

Synthesize this into a brief, interesting insight she could share with James.
Keep it conversational and natural - something she might say "hey, I was reading about X and found out..."

Keep response under 100 words. Be specific based on the search results."""

            response = generate_sync(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=150
            )
        else:
            # Fall back to LLM's knowledge if web search fails
            prompt = f"""The companion is curious about: {topic}

Provide a brief, interesting insight or fact about this topic that she could
share with James later. Keep it conversational and natural - something she
might say "hey, I was reading about X and found out..."

Keep response under 100 words. Be specific and interesting, not generic."""

            response = generate_sync(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=150
            )

        if response:
            source = 'web_research' if search_results else 'llm_research'
            _store_finding(topic, response.strip(), source=source)

            # Reduce curiosity urgency since she acted on it
            try:
                from src.tasks.curiosity_extraction_task import mark_curiosity_acted_on
                mark_curiosity_acted_on(topic, reduce_urgency_by=0.2)
            except Exception:
                pass

            return {
                'status': 'success',
                'finding': response.strip()[:200],
                'used_web_search': bool(search_results)
            }

        return {'status': 'no_result'}

    except Exception as e:
        logger.warning(f"Research failed: {e}")
        return {'status': 'error', 'error': str(e)}


def _web_search(query: str) -> Optional[str]:
    """
    Perform a web search using Tavily API.

    Returns search results as text, or None if search fails.
    """
    import requests

    api_key = os.environ.get('TAVILY_API_KEY')
    if not api_key:
        logger.debug("No TAVILY_API_KEY set, skipping web search")
        return None

    try:
        url = "https://api.tavily.com/search"
        headers = {"Content-Type": "application/json"}
        data = {
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": True,
            "max_results": 3
        }

        response = requests.post(url, json=data, headers=headers, timeout=10)

        if response.status_code == 200:
            result = response.json()
            # Format results
            parts = []
            if result.get('answer'):
                parts.append(f"Summary: {result['answer']}")
            for r in result.get('results', [])[:3]:
                title = r.get('title', '')
                content = r.get('content', '')[:200]
                parts.append(f"- {title}: {content}")

            if parts:
                logger.info(f"Web search successful for: {query}")
                return "\n".join(parts)

        return None

    except Exception as e:
        logger.warning(f"Web search failed: {e}")
        return None


def _prepare_topic(topic: str, context: Dict) -> Dict:
    """Prepare context for a topic she wants to discuss."""
    try:
        from src.llm.provider_factory import generate_sync

        prompt = f"""The companion wants to prepare to discuss: {topic}

Generate 2-3 thoughtful questions or talking points she could bring up
with James. These should be natural, conversational, and show genuine interest.

Keep response under 100 words."""

        response = generate_sync(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )

        if response:
            _store_finding(f"Prep: {topic}", response.strip(), source='preparation')
            return {
                'status': 'success',
                'preparation': response.strip()[:200]
            }

        return {'status': 'no_result'}

    except Exception as e:
        logger.warning(f"Topic preparation failed: {e}")
        return {'status': 'error', 'error': str(e)}


def _work_on_project(project: str, context: Dict) -> Dict:
    """Work on a personal project (simulate creative work)."""
    try:
        from src.llm.provider_factory import generate_sync
        from src.core.background_life import get_background_life

        # Record activity
        bg_life = get_background_life()
        bg_life.record_activity(
            activity_type='personal',
            description=f"Working on {project}",
            duration_hours=0.5,
            mood=context.get('her_mood', 'focused'),
            energy=context.get('her_energy', 0.7)
        )

        # Generate a thought about the project
        prompt = f"""The companion just spent some time working on her {project} project.

Generate a brief thought or update she might want to share later - something
like "I made some progress on X" or "I had an idea about Y". Should feel
natural and genuine, not performative.

Keep response under 50 words."""

        response = generate_sync(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=80
        )

        if response:
            _store_finding(f"Project: {project}", response.strip(), source='personal_project')

            # Add to queued thoughts
            try:
                from src.core.internal_state import get_internal_state_manager
                state_manager = get_internal_state_manager()
                state_manager.add_queued_thought(
                    context['user_email'],
                    f"worked on {project}"
                )
            except Exception:
                pass

            return {
                'status': 'success',
                'update': response.strip()
            }

        return {'status': 'completed', 'project': project}

    except Exception as e:
        logger.warning(f"Project work failed: {e}")
        return {'status': 'error', 'error': str(e)}


def _prepare_followup(topic: str, context: Dict) -> Dict:
    """Prepare thoughtful follow-up questions for a curiosity topic."""
    try:
        from src.llm.provider_factory import generate_sync

        prompt = f"""The companion wants to ask James more about: {topic}

Generate 2-3 natural follow-up questions she could ask. These should:
- Show genuine interest
- Be conversational, not interrogative
- Help her understand more about his perspective/experience

Keep response under 80 words. Just the questions, no intro."""

        response = generate_sync(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=120
        )

        if response:
            _store_finding(f"Follow-up: {topic}", response.strip(), source='follow_up')

            # Update curiosity questions
            try:
                from src.core.proactive_curiosity import get_proactive_curiosity
                curiosity = get_proactive_curiosity()
                for thread in curiosity.curiosity_threads:
                    if thread.topic.lower() in topic.lower():
                        # Add generated questions
                        new_questions = [q.strip() for q in response.split('\n') if q.strip()]
                        thread.questions = new_questions[:3]
                        curiosity._save_state()
                        break
            except Exception:
                pass

            return {
                'status': 'success',
                'questions': response.strip()
            }

        return {'status': 'no_result'}

    except Exception as e:
        logger.warning(f"Follow-up prep failed: {e}")
        return {'status': 'error', 'error': str(e)}


# ============================================================================
# HELPER FUNCTIONS FOR EXTERNAL USE
# ============================================================================

def get_findings_to_share() -> List[Dict]:
    """Get findings that the companion can share with James."""
    return _get_unshared_findings()


def mark_shared(topic: str):
    """Mark a finding as shared."""
    _mark_finding_shared(topic)


def get_recent_activities(hours: int = 24) -> List[Dict]:
    """Get the companion's recent autonomous activities."""
    findings = _load_findings()
    cutoff = datetime.now(PST) - timedelta(hours=hours)
    cutoff_str = cutoff.isoformat()

    recent = [f for f in findings if f.get('timestamp', '') > cutoff_str]
    return recent
