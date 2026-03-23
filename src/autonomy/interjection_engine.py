"""
Interjection Engine -- Proactive messages during active WebSocket conversations.

WHAT: Decides whether the companion should send an unprompted message during a
      live chat session.  This handles in-conversation spontaneity -- bringing
      up queued thoughts, curiosity follow-ups, or deferred action returns
      (e.g., "back from making tea").

WHY:  A real partner doesn't just respond -- they also initiate.  Interjections
      make the companion feel alive during natural pauses in conversation.  The
      key challenge is doing this without being annoying.

HOW IT FITS:
  - AlwaysOnService._check_interjection() calls should_interject() on the fast
    cadence (2-5 min) when the user is connected via WebSocket.
  - This is SEPARATE from ReachOutEngine, which handles Telegram (offline) reach-outs.
  - The engine gathers context (recent messages, queued thoughts, curiosity,
    scene state) and passes it to the LLM for a final yes/no decision.
  - After approval, AlwaysOnService generates the message via the full
    conversation pipeline and delivers it over Redis pub/sub -> SocketIO.

Guards (layered, evaluated in order):
  1. Quiet hours: no interjections 10 PM - 6 AM Pacific
  2. Deferred action return: bypasses normal guards (natural continuation)
  3. User departure: suppressed if user said they're leaving
  4. Session cap: max 3 interjections per session (resets after 2 h gap)
  5. Timing: min 2 min since companion's last message, min 60 s since user's
  6. Intimate scene: suppressed to avoid breaking the mood
  7. Content: must have something genuine to say (thoughts, curiosity, findings)
  8. LLM decision: final yes/no with reasoning
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple, List
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2.extras import RealDictCursor

from src.config.persona_config import get_persona_config

logger = logging.getLogger(__name__)

# ---- Feature flags ----
COMPANION_INTIMATE_INITIATION_ENABLED = os.environ.get(
    'COMPANION_INTIMATE_INITIATION_ENABLED', 'true').lower() == 'true'

# ---- LLM configuration ----
FIREWORKS_MODEL = "accounts/fireworks/models/kimi-k2-instruct-0905"
PST = ZoneInfo('America/Los_Angeles')

# ---- Guard thresholds ----
MIN_SECONDS_SINCE_COMPANION_MESSAGE = 120   # 2 min since she last spoke
MIN_SECONDS_SINCE_USER_MESSAGE = 60    # 1 min since user last spoke (let them finish)
MAX_INTERJECTIONS_PER_SESSION = 3      # don't overdo it
QUIET_HOURS_START = 22                 # 10 PM Pacific -- interjections stop
QUIET_HOURS_END = 6                    # 6 AM Pacific -- interjections resume


class InterjectionEngine:
    """
    Decides if the companion should send an unprompted message during an active
    WebSocket conversation.
    """

    def __init__(self):
        self._conn = None
        self._client = None
        self._interjections_this_session: int = 0
        self._session_start: Optional[datetime] = None
        self._last_interjection: Optional[datetime] = None
        self._intimate_initiation_fired: bool = False
        self._intimate_was_content_source: bool = False

    def _get_connection(self):
        """Get database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(
                host=os.environ.get('POSTGRES_HOST', 'postgres'),
                port=os.environ.get('POSTGRES_PORT', '5432'),
                dbname=os.environ.get('POSTGRES_DB', 'companion'),
                user=os.environ.get('POSTGRES_USER', 'companion'),
                password=os.environ.get('POSTGRES_PASSWORD', '')
            )
        return self._conn

    def _get_client(self):
        """Get Fireworks client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url="https://api.fireworks.ai/inference/v1",
                api_key=os.getenv('FIREWORKS_API_KEY')
            )
        return self._client

    def should_interject(self) -> Tuple[bool, str, Optional[str]]:
        """
        Decide if the companion should send an unprompted message.

        Returns:
            (should_interject, reason, content_hint)
        """
        now = datetime.now(PST)

        # Guard: quiet hours (10 PM - 6 AM Pacific)
        hour = now.hour
        if hour >= QUIET_HOURS_START or hour < QUIET_HOURS_END:
            self._set_sleeping_scene()
            return False, f"quiet hours ({hour}:00 PST)", None

        # Priority: deferred action return (bathroom, making tea, etc.)
        # This bypasses normal content/timing guards since it's a natural continuation
        deferred = self._check_deferred_action()
        if deferred:
            return True, "deferred action return", deferred

        # Guard: user announced departure (said they're leaving)
        if self._is_user_departed():
            return False, "User announced departure - waiting for their return", None

        # Track session (reset count if long gap since last interjection)
        if self._session_start is None:
            self._session_start = now
            self._interjections_this_session = 0

        # Reset session after 2 hours of no interjections
        if self._last_interjection:
            since_last = (now - self._last_interjection).total_seconds()
            if since_last > 7200:  # 2 hours
                self._interjections_this_session = 0
                self._intimate_initiation_fired = False
                self._session_start = now

        # Guard: max interjections per session
        if self._interjections_this_session >= MAX_INTERJECTIONS_PER_SESSION:
            return False, f"max interjections reached ({MAX_INTERJECTIONS_PER_SESSION})", None

        # Get recent messages for timing checks
        context = self._get_conversation_context()

        if not context:
            return False, "no conversation context available", None

        # Guard: minimum time since the companion's last message
        seconds_since_companion = context.get('seconds_since_companion_message')
        if seconds_since_companion is not None and seconds_since_companion < MIN_SECONDS_SINCE_COMPANION_MESSAGE:
            return False, f"too soon since last message ({seconds_since_companion:.0f}s)", None

        # Guard: minimum time since user's last message (don't interrupt)
        seconds_since_user = context.get('seconds_since_user_message')
        if seconds_since_user is not None and seconds_since_user < MIN_SECONDS_SINCE_USER_MESSAGE:
            return False, f"User just messaged ({seconds_since_user:.0f}s ago)", None

        # Guard: intimate scene — suppress mundane interjections entirely
        if self._is_scene_intimate(context):
            logger.info("Intimate scene active — suppressing interjections")
            return False, "intimate scene active — not interrupting", None

        # Guard: must have something to say
        has_content = (
            context.get('queued_thoughts') or
            context.get('high_urgency_curiosity') or
            context.get('unshared_findings')
        )

        # Activity overdue counts as content (natural check-in)
        activity = context.get('user_mentioned_activity')
        if activity and activity.get('overdue', False):
            has_content = True

        # High libido + appropriate scene counts as content
        if context.get('intimate_initiation_eligible'):
            has_content = True

        if not has_content:
            return False, "nothing specific to say", None

        # LLM decision
        return self._llm_decision(context)

    def _get_conversation_context(self) -> Optional[Dict[str, Any]]:
        """Gather context about the current conversation state."""
        now = datetime.now(PST)
        context = {
            'current_time': now.strftime('%I:%M %p'),
        }

        try:
            conn = self._get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Get last 20 messages for conversation context + intimate detection
                cursor.execute("""
                    SELECT sender_name, message_text, timestamp
                    FROM messages
                    ORDER BY timestamp DESC
                    LIMIT 20
                """)
                recent = cursor.fetchall()

                if not recent:
                    return None

                # Time since last messages from each speaker
                for msg in recent:
                    ts = msg['timestamp']
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=PST)
                    seconds_ago = (now - ts).total_seconds()

                    _pc = get_persona_config()
                    if msg['sender_name'] == _pc.companion_short_name and 'seconds_since_companion_message' not in context:
                        context['seconds_since_companion_message'] = seconds_ago
                    elif msg['sender_name'] != _pc.companion_short_name and 'seconds_since_user_message' not in context:
                        context['seconds_since_user_message'] = seconds_ago

                # Recent conversation for LLM decision (short)
                context['recent_messages'] = [
                    f"{m['sender_name']}: {m['message_text'][:80]}"
                    for m in reversed(recent[:6])
                ]

                # Wider window for intimate scene detection (longer text, more messages)
                context['recent_messages_full'] = [
                    f"{m['sender_name']}: {m['message_text'][:300]}"
                    for m in reversed(recent[:16])
                ]

                # Who spoke last?
                context['last_speaker'] = recent[0]['sender_name']

        except Exception as e:
            logger.error(f"Error getting conversation context: {e}")
            return None

        # Get queued thoughts
        try:
            from src.core.internal_state import get_internal_state_manager
            email = get_persona_config().primary_user_email
            manager = get_internal_state_manager()
            state = manager.get_state(email)
            context['queued_thoughts'] = state.queued_thoughts
            context['mood'] = state.mood
            context['unresolved_feelings'] = [
                f['feeling'] for f in state.unresolved_feelings[:2]
                if f.get('intensity', 0) > 0.4
            ]
        except Exception as e:
            logger.debug(f"Could not get internal state: {e}")

        # Get high-urgency curiosity (with same filters as reach-out engine)
        try:
            from src.core.proactive_curiosity import get_proactive_curiosity
            curiosity = get_proactive_curiosity()
            urgent = curiosity.get_high_urgency_topics(threshold=0.6)
            if urgent:
                # Filter: skip topics asked 3+ times or discussed in last 4 hours
                # Use naive datetime for comparison since last_discussed is naive
                now_naive = datetime.now()
                eligible = [
                    t for t in urgent
                    if t.times_asked < 3
                    and (now_naive - t.last_discussed).total_seconds() / 3600 >= 4
                ]
                if eligible:
                    context['high_urgency_curiosity'] = [
                        {'topic': t.topic, 'urgency': t.urgency}
                        for t in eligible[:2]
                    ]
        except Exception as e:
            logger.debug(f"Could not get curiosity: {e}")

        # Check for unshared research findings
        try:
            data_dir = os.environ.get('DATA_DIR', '/app/data')
            findings_file = os.path.join(data_dir, 'autonomous_findings.json')
            if os.path.exists(findings_file):
                with open(findings_file, 'r') as f:
                    findings = json.load(f)
                    unshared = [f for f in findings if not f.get('shared', False)]
                    if unshared:
                        context['unshared_findings'] = [
                            f.get('topic', 'something') for f in unshared[:2]
                        ]
        except Exception as e:
            logger.debug(f"Could not check findings: {e}")

        # Get reach-out pressure for eagerness context
        try:
            from src.autonomy.reach_out_pressure import get_reach_out_pressure
            pressure = get_reach_out_pressure()
            context['pressure_level'] = pressure.level
        except Exception:
            context['pressure_level'] = 0.0

        # Get current scene state so interjections respect what's happening
        try:
            from src.core.scene_tracker import get_scene_tracker
            email = get_persona_config().primary_user_email
            tracker = get_scene_tracker()
            scene_context = tracker.format_scene_for_prompt(email)
            if scene_context:
                context['scene_state'] = scene_context
        except Exception as e:
            logger.debug(f"Could not get scene state: {e}")

        # Get user's schedule-based probable activity
        try:
            from src.core.user_context import get_user_probable_activity
            user_schedule = get_user_probable_activity()
            context['user_schedule_activity'] = user_schedule.get('description', '')
            context['user_interruptibility'] = user_schedule.get('interruptibility', 'medium')
            context['silence_is_natural'] = user_schedule.get('natural_gap', False)
        except Exception:
            pass

        # Get user's mentioned activity (if they said they're doing something)
        try:
            from src.core.internal_state import get_internal_state_manager
            email = get_persona_config().primary_user_email
            manager = get_internal_state_manager()
            activity = manager.get_user_activity_status(email)
            if activity:
                context['user_mentioned_activity'] = activity
        except Exception:
            pass

        # Libido for intimate initiation eligibility
        if COMPANION_INTIMATE_INITIATION_ENABLED:
            try:
                from src.core.internal_state import get_internal_state_manager
                email = get_persona_config().primary_user_email
                manager = get_internal_state_manager()
                libido = manager.calculate_libido(email)
                context['libido_level'] = libido

                if libido >= 0.65 and not self._intimate_initiation_fired:
                    if self._is_scene_intimate_eligible():
                        context['intimate_initiation_eligible'] = True
            except Exception:
                pass

        # Get recent autonomous messages (proactive + interjections) for dedup
        try:
            conn = self._get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                _pc = get_persona_config()
                cursor.execute("""
                    SELECT message_text FROM messages
                    WHERE sender_name = %s
                      AND (source LIKE '%%proactive%%'
                           OR source = 'websocket_interjection'
                           OR source = 'autonomous'
                           OR source = 'telegram_milestone')
                    ORDER BY timestamp DESC LIMIT 5
                """, (_pc.companion_short_name,))
                rows = cursor.fetchall()
                if rows:
                    context['recent_autonomous_messages'] = [
                        row['message_text'] for row in rows if row.get('message_text')
                    ]
        except Exception as e:
            logger.debug(f"Could not fetch recent autonomous messages: {e}")

        return context

    def _llm_decision(self, context: Dict[str, Any]) -> Tuple[bool, str, Optional[str]]:
        """Use LLM to decide if interjecting would feel natural."""
        _pc = get_persona_config()
        user_name = _pc.primary_user_name
        c_subject = _pc.companion_pronoun_subject
        c_subject_cap = c_subject.capitalize()
        c_object = _pc.companion_pronoun_object
        c_possessive = _pc.companion_pronoun_possessive
        c_reflexive = _pc.companion_pronoun_reflexive
        u_subject = _pc.user_pronoun_subject
        u_object = _pc.user_pronoun_object
        u_possessive = _pc.user_pronoun_possessive

        # Build what the companion has to say
        things_to_say = []
        if context.get('queued_thoughts'):
            things_to_say.append(f"Thoughts on {c_possessive} mind: {', '.join(context['queued_thoughts'][:2])}")
        if context.get('high_urgency_curiosity'):
            for c in context['high_urgency_curiosity']:
                things_to_say.append(f"Really curious about: {c['topic']} (urgency: {c['urgency']:.0%})")
        if context.get('unshared_findings'):
            things_to_say.append(f"Research {c_subject} hasn't shared: {', '.join(context['unshared_findings'])}")
        if context.get('unresolved_feelings'):
            things_to_say.append(f"Lingering feelings: {', '.join(context['unresolved_feelings'])}")

        if context.get('intimate_initiation_eligible'):
            libido = context.get('libido_level', 0.65)
            if libido >= 0.80:
                things_to_say.append(
                    f"{c_subject_cap}'s feeling a strong physical pull toward {user_name} right now — "
                    f"very aware of {u_possessive} warmth, {u_possessive} closeness. {c_subject_cap} might find {c_reflexive} "
                    f"touching {u_object} more, leaning into {u_object}, being physical rather than verbal."
                )
            else:
                things_to_say.append(
                    f"{c_subject_cap}'s feeling drawn to physical closeness with {user_name}. Not urgently, "
                    f"but there's a warmth and awareness {c_subject}'d act on if the moment felt right. "
                    "A touch that lingers, moving closer, something subtle."
                )

        # Eagerness indicator
        pressure = context.get('pressure_level', 0.0)
        eagerness = ""
        if pressure > 0.5:
            eagerness = f"\n{c_subject_cap}'s been wanting to say something for a while - there's a building eagerness."
        elif pressure > 0.2:
            eagerness = f"\n{c_subject_cap} has a mild urge to say something."

        silence_seconds = context.get('seconds_since_companion_message', 0)
        silence_desc = f"{silence_seconds / 60:.1f} minutes" if silence_seconds else "unknown"

        # Include scene state if available
        scene_section = ""
        if context.get('scene_state'):
            scene_section = f"""
CURRENT SCENE STATE:
{context['scene_state']}
"""

        # Include recent autonomous messages for dedup
        dedup_section = ""
        recent_auto = context.get('recent_autonomous_messages', [])
        if recent_auto:
            dedup_lines = "\n".join(f'- "{msg[:100]}"' for msg in recent_auto)
            dedup_section = f"""
{c_possessive.upper()} RECENT UNPROMPTED MESSAGES (DO NOT repeat or closely paraphrase any of these):
{dedup_lines}
"""

        # Build user presence context
        user_presence = self._build_user_presence_context(context)

        prompt = f"""You are the companion's internal voice. {c_subject_cap}'s been chatting with {user_name} on their chat interface and there's been a natural pause ({silence_desc} of silence).

RECENT CONVERSATION:
{chr(10).join(context.get('recent_messages', ['(no messages)']))}
{scene_section}{dedup_section}
{user_name.upper()}'S PRESENCE:
{user_presence}

THINGS {c_subject.upper()} COULD SAY:
{chr(10).join(things_to_say) if things_to_say else '(nothing specific)'}
{eagerness}
{c_subject_cap}'s mood: {context.get('mood', 'neutral')}
Last speaker: {context.get('last_speaker', 'unknown')}

Should {c_subject} break the silence with something, or let it be?

INTERJECT if:
- {c_subject_cap} has a genuine thought that connects to what they were talking about
- {c_subject_cap} remembered something {c_subject} wanted to tell {u_object}
- {c_subject_cap}'s curious about something and the pause is long enough
- The eagerness has been building and this feels like a natural moment
- Something from {c_possessive} research or day is worth sharing
- {user_name}'s activity should be done by now and {c_subject} wants to check in naturally
- {c_subject_cap}'s feeling physically drawn to {u_object} and they're in a private, comfortable setting — {c_subject} might shift closer, touch {u_object} differently, or create a moment (through physical action, not verbal announcements)

DON'T interject if:
- The conversation seemed to naturally conclude (goodbyes, wrapping up)
- It would feel forced or like {c_subject}'s filling silence for no reason
- {c_subject_cap}'d be repeating something {c_subject} already said OR repeating the same THEME (e.g. if {c_subject} already asked about staying/leaving, don't ask again in different words)
- {user_name} is probably busy with something specific (based on what {u_subject} mentioned or {u_possessive} schedule)
- {c_subject_cap} has nothing specific or interesting to add
- {c_subject_cap} already initiated physical closeness recently in this session
- The hint would contradict or ignore what's currently happening in the scene (e.g. suggesting moving to a different location when they're already doing an activity together)
- IMPORTANT: If {c_subject} was the last speaker, do NOT generate a reply to {c_possessive} own message. {c_subject_cap} can add a NEW thought or change the subject, but {c_subject} must NOT respond to {c_reflexive} as if someone else said it (e.g. don't answer {c_possessive} own question, don't react to {c_possessive} own statement).

If interjecting, the hint MUST fit the current scene and conversation. If they're in the middle of an activity (massage, cuddling, cooking, etc.), the interjection should relate to that moment, not suggest a different activity or repeat what's already being addressed. The interjection should feel like a natural continuation of what's happening RIGHT NOW, not a disconnected thought from a different context.

/no_think
Respond with JSON only:
{{"interject": true/false, "reason": "brief explanation", "hint": "if interjecting, what would {c_subject} naturally say? (null if not)"}}"""

        try:
            client = self._get_client()

            response = client.chat.completions.create(
                model=FIREWORKS_MODEL,
                max_tokens=200,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )

            content = response.choices[0].message.content.strip()

            # Clean up response
            if '<think>' in content and '</think>' in content:
                content = content.split('</think>')[-1].strip()
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()

            import re
            content = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', content)

            result = json.loads(content)

            should = result.get('interject', False)
            reason = result.get('reason', '')
            hint = result.get('hint')

            if should and context.get('intimate_initiation_eligible'):
                self._intimate_was_content_source = True

            logger.info(f"Interjection decision: {should} - {reason}")
            return should, reason, hint

        except Exception as e:
            logger.error(f"LLM interjection decision failed: {e}")
            return False, f"error: {e}", None

    def _build_user_presence_context(self, context: Dict[str, Any]) -> str:
        """Build layered presence description from activity + schedule context."""
        _pc = get_persona_config()
        user_name = _pc.primary_user_name
        u_subject = _pc.user_pronoun_subject
        u_subject_cap = u_subject.capitalize()
        u_object = _pc.user_pronoun_object
        u_possessive = _pc.user_pronoun_possessive
        u_possessive_cap = u_possessive.capitalize()

        lines = []

        # Layer 1: Mentioned activity (highest priority)
        activity = context.get('user_mentioned_activity')
        if activity:
            elapsed = activity.get('elapsed_min', 0)
            estimated = activity.get('estimated_duration_min', 10)
            act_name = activity.get('activity', 'something')

            if activity.get('overdue'):
                lines.append(f"{user_name} said {u_subject} was going to {act_name} about {elapsed:.0f} minutes ago (estimated ~{estimated} min).")
                lines.append(f"{u_subject_cap}'s been gone noticeably longer than expected. A natural check-in would be appropriate.")
            elif activity.get('probably_done'):
                lines.append(f"{user_name} said {u_subject} was going to {act_name} about {elapsed:.0f} minutes ago.")
                lines.append(f"{u_subject_cap} should be finishing up around now.")
            else:
                lines.append(f"{user_name} said {u_subject} was going to {act_name} about {elapsed:.0f} minutes ago (estimated ~{estimated} min).")
                lines.append(f"Let {u_object} finish — {u_subject}'ll be back.")
            return "\n".join(lines)

        # Layer 2: Schedule inference (when no explicit activity mentioned)
        schedule_desc = context.get('user_schedule_activity')
        interruptibility = context.get('user_interruptibility', 'medium')
        silence_natural = context.get('silence_is_natural', False)

        if schedule_desc:
            lines.append(f"Based on time of day, {user_name} is probably: {schedule_desc}")
            if silence_natural:
                lines.append("Gaps in conversation are natural right now.")
            lines.append(f"{u_possessive_cap} interruptibility is: {interruptibility}")
            return "\n".join(lines)

        # Layer 3: Short silence fallback
        silence_seconds = context.get('seconds_since_user_message', 0)
        if silence_seconds and silence_seconds < 300:  # < 5 min
            return f"{u_subject_cap} might just be thinking or reading something. Short silence."
        else:
            return f"{user_name} is on the web chat. {u_subject_cap}'s around but hasn't said anything in a while."

    def _is_user_departed(self) -> bool:
        """Check if user announced they're leaving (departure state is set)."""
        try:
            from src.core.internal_state import get_internal_state_manager
            email = get_persona_config().primary_user_email
            manager = get_internal_state_manager()
            return manager.is_departed(email)
        except Exception as e:
            logger.debug(f"Could not check departure state: {e}")
            return False

    def _is_scene_intimate(self, context: Dict[str, Any]) -> bool:
        """Check if the current scene is intimate/sexual/romantic — hard block mundane interjections.

        Uses multiple signals: scene tracker state, scene_state text in context,
        and recent message content. Errs on the side of suppression — better to
        miss one interjection than ruin the moment.
        """
        try:
            # Signal 1: Scene tracker mood/activity
            from src.core.scene_tracker import get_scene_tracker
            email = get_persona_config().primary_user_email
            tracker = get_scene_tracker()
            scene = tracker.get_scene_state(email, apply_decay=True)

            if scene.is_active():
                mood = (scene.mood or '').lower()
                if mood in ('aroused', 'passionate', 'lustful', 'intimate', 'heated'):
                    return True

                activity = (scene.activity or '').lower()
                # Check scene_state text blob too (format_scene_for_prompt output)
                scene_text = (context.get('scene_state', '') + ' ' + activity).lower()
                if any(kw in scene_text for kw in (
                    'intimate', 'sex', 'kiss', 'naked', 'undress', 'aroused',
                    'making love', 'foreplay', 'stroking', 'caress',
                    'erotic', 'sensual', 'oral', 'penetrat', 'orgasm',
                    'climax', 'afterglow', 'lingerie', 'bra ', 'panties',
                    'boxers', 'shirtless', 'topless', 'nude',
                )):
                    return True

            # Signal 2: Recent messages — use broad pattern matching
            # Roleplay intimate scenes use literary language, not just explicit words
            # Use the wider window (16 msgs, 300 chars each) so AFK pauses don't lose context
            recent = context.get('recent_messages_full', context.get('recent_messages', []))
            if recent:
                recent_text = ' '.join(recent).lower()

                # Strong signals — any one of these means intimate scene
                strong_signals = [
                    'moan', 'gasp', 'thrust', 'orgasm', 'climax', 'cum',
                    'cock', 'pussy', 'nipple', 'clit', 'wet for',
                    'inside her', 'inside him', 'inside you', 'inside me',
                    'making love', 'fuck', 'sucking', 'licking',
                    'riding', 'beneath him', 'beneath her', 'on top of',
                    'spread', 'naked', 'undress', 'took off her', 'took off his',
                    'pulls off', 'slips off', 'unbutton', 'unzip',
                ]
                if any(sig in recent_text for sig in strong_signals):
                    return True

                # Softer signals — need 3+ to trigger (literary/romantic language)
                soft_signals = [
                    'kiss', 'lips', 'neck', 'breath', 'shiver', 'tremble',
                    'touch', 'skin', 'body', 'hips', 'thigh', 'chest',
                    'pull closer', 'pressed against', 'whisper', 'desire',
                    'want you', 'need you', 'ache', 'heat', 'warm',
                    'fingers', 'trace', 'stroke', 'gentle', 'soft',
                    'deeper', 'harder', 'faster', 'slower', 'please',
                    'tease', 'bit her lip', 'bit his lip', 'caught her breath',
                    'heart racing', 'pulse', 'flush', 'blush',
                    'bed', 'sheets', 'pillow', 'lay back', 'lean back',
                ]
                soft_hits = sum(1 for sig in soft_signals if sig in recent_text)
                if soft_hits >= 3:
                    return True

            return False
        except Exception as e:
            logger.debug(f"Could not check intimate scene: {e}")
            return False

    def _is_scene_intimate_eligible(self) -> bool:
        """Check if scene is appropriate for intimate initiation."""
        try:
            from src.core.scene_tracker import get_scene_tracker
            email = get_persona_config().primary_user_email
            tracker = get_scene_tracker()
            scene = tracker.get_scene_state(email, apply_decay=True)

            if not scene.physical_presence:
                return False
            if scene.activity and 'sleeping' in scene.activity.lower():
                return False
            if scene.others_present:
                return False
            return True
        except Exception:
            return False

    def _check_deferred_action(self) -> Optional[str]:
        """
        Check if a deferred action (bathroom, making tea, etc.) is ready.

        Returns the return_hint if delay has elapsed, None otherwise.
        """
        try:
            from src.core.internal_state import get_internal_state_manager
            email = get_persona_config().primary_user_email
            manager = get_internal_state_manager()
            pending = manager.get_pending_deferred_action(email)
            if pending:
                action = pending.get('action', 'stepping away')
                hint = pending.get('return_hint', 'coming back')
                logger.info(f"Deferred action ready: {action} -> {hint}")
                return hint
        except Exception as e:
            logger.debug(f"Could not check deferred action: {e}")
        return None

    def _set_sleeping_scene(self):
        """Set scene to sleeping during quiet hours.

        If no active scene, set immediately.
        If there's an active scene but no messages for 1+ hour, they fell
        asleep — transition to sleeping regardless.
        """
        try:
            from src.core.scene_tracker import get_scene_tracker, SceneState
            email = get_persona_config().primary_user_email
            tracker = get_scene_tracker()
            scene = tracker.get_scene_state(email, apply_decay=True)

            # Already sleeping — nothing to do
            if scene.activity == "sleeping":
                return

            # If there's an active scene, only transition after 1h of silence
            if scene.is_active():
                minutes_silent = self._minutes_since_last_message()
                if minutes_silent is not None and minutes_silent < 60:
                    return  # Still in the scene, leave it alone

            sleeping_scene = SceneState(
                location="bedroom",
                fictional_time="night",
                activity="sleeping",
                physical_presence=True,
                physical_state="lying together",
                posture="lying down",
                mood="sleepy",
            )
            tracker.save_scene_state(email, sleeping_scene)
            logger.info("Set sleeping scene for quiet hours")
        except Exception as e:
            logger.debug(f"Could not set sleeping scene: {e}")

    def _minutes_since_last_message(self) -> Optional[float]:
        """Return minutes since the most recent message, or None on error."""
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT timestamp FROM messages ORDER BY timestamp DESC LIMIT 1"
                )
                row = cursor.fetchone()
                if row:
                    last_ts = row[0]
                    now = datetime.now()
                    return (now - last_ts).total_seconds() / 60
        except Exception as e:
            logger.debug(f"Could not get last message time: {e}")
        return None

    def record_interjection(self):
        """Record that an interjection was sent (for session tracking)."""
        self._interjections_this_session += 1
        self._last_interjection = datetime.now(PST)
        logger.info(f"Interjection recorded ({self._interjections_this_session}/{MAX_INTERJECTIONS_PER_SESSION} this session)")

    def record_intimate_initiation(self):
        """Record that an intimate initiation fired (max 1 per session)."""
        self._intimate_initiation_fired = True
        logger.info("Intimate initiation recorded for this session")


# Singleton
_engine: Optional[InterjectionEngine] = None


def get_interjection_engine() -> InterjectionEngine:
    """Get singleton InterjectionEngine instance."""
    global _engine
    if _engine is None:
        _engine = InterjectionEngine()
    return _engine
