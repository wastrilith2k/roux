#!/usr/bin/env python3
"""
Simulation Runner — Companion Framework

WHAT: Runs a multi-agent relationship simulation where two AI companions (e.g., Kai
      and Mira) talk to each other over simulated weeks, developing a relationship.
WHY:  Developing and testing the framework's autonomous subsystems (memory, opinions,
      curiosity, episodes, value inference, etc.) requires relationship evolution that
      takes weeks of real interaction. This script compresses weeks into minutes by
      using SimulationClock to control time and generating 2-3 conversations per
      simulated day.
HOW:  SimulationRunner manages the simulation lifecycle:
      - Each day: morning schedule -> 2-3 conversations -> end-of-day processing
      - Each conversation: 3-8 message exchanges between initiator and responder
      - End of day: episode extraction, opinion formation, curiosity/confidence decay
      - Weekly: reflection tasks
      - Monthly: value inference
      After each week, a PostgreSQL checkpoint is saved for review and rollback.
      The --rollback flag restores a previous checkpoint by dropping and recreating
      the database from the SQL dump.

Usage:
    python scripts/simulate_relationship.py --week 1
    python scripts/simulate_relationship.py --week 2
    python scripts/simulate_relationship.py --rollback week1
    python scripts/simulate_relationship.py --summary
"""

import argparse
import json
import logging
import os
import random
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

PST = ZoneInfo('America/Los_Angeles')

# ---------------------------------------------------------------------------
# Event emitter — publishes simulation events to Redis for the dashboard
# ---------------------------------------------------------------------------

class SimulationEventEmitter:
    """Publishes simulation events to Redis pub/sub for live observation."""

    def __init__(self):
        self.redis = None
        self.channel = 'simulation_events'
        try:
            import redis
            self.redis = redis.Redis(
                host=os.environ.get('REDIS_HOST', 'redis'),
                port=int(os.environ.get('REDIS_PORT', 6379)),
                decode_responses=True
            )
            self.redis.ping()
        except Exception as e:
            print(f"Warning: Redis not available for event emission: {e}")
            self.redis = None

    def emit(self, event_type: str, data: dict):
        """Publish an event to the simulation_events channel."""
        if not self.redis:
            return
        try:
            self.redis.publish(self.channel, json.dumps({
                'type': event_type,
                'data': data,
                'timestamp': datetime.now(PST).isoformat(),
            }))
        except Exception:
            pass  # Non-critical — dashboard just won't update

    def set_status(self, status: dict):
        """Set the simulation:status Redis key for polling."""
        if not self.redis:
            return
        try:
            self.redis.set('simulation:status', json.dumps(status))
        except Exception:
            pass


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/simulation.log', mode='a'),
    ]
)
logger = logging.getLogger(__name__)


def find_project_root() -> Path:
    candidates = [Path(__file__).parent.parent, Path('/app'), Path.cwd()]
    for p in candidates:
        if (p / 'data').exists():
            return p
    raise RuntimeError("Cannot find project root")


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

class SimulationRunner:
    """Runs a week-by-week simulation of two companions interacting.

    Uses SimulationClock to control time globally. All framework modules
    that call clock.now() will see the simulated time, making memories,
    schedules, and temporal context work correctly.
    """

    def __init__(self, companions: list, start_day: int = 0, with_analysis: bool = False,
                 config_path: str = None, full_stack: bool = False,
                 api_url: str = 'http://localhost:5001', celery_wait: int = 5):
        """
        Args:
            companions: list of companion_ids, e.g. ["kai", "mira"]
            start_day: resume from this day (0 = beginning)
            with_analysis: run MessageAnalyzer after each message (doubles LLM calls)
            config_path: path to simulation config YAML
            full_stack: route messages through the HTTP API for realistic cost tracking
            api_url: base URL for the HTTP API (default: http://localhost:5001)
            celery_wait: seconds to wait after each conversation for Celery tasks (default: 5)
        """
        from src.core.clock import SimulationClock, set_clock
        self.project_root = find_project_root()
        sys.path.insert(0, str(self.project_root))

        self.clock = SimulationClock(start=datetime(2026, 1, 1, 7, 0, tzinfo=PST))
        set_clock(self.clock)

        self.companions = companions
        self.start_day = start_day
        self.with_analysis = with_analysis
        self._config_path = config_path
        self.full_stack = full_stack
        self.api_url = api_url.rstrip('/')
        self.celery_wait = celery_wait
        self.checkpoint_dir = self.project_root / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.emitter = SimulationEventEmitter()

        # Advance clock if resuming
        if start_day > 0:
            self.clock.advance(days=start_day)

        self.emitter.set_status({
            'status': 'initialized',
            'companions': companions,
            'start_day': start_day,
            'clock_time': self.clock.now().isoformat(),
        })

        # Load config from external YAML files
        self._load_simulation_config()

        # Ensure Postgres schema and tables exist (bypasses normal web_chat init path)
        self._ensure_schema()

        # Ensure user profiles exist for simulation emails (required by FK constraint)
        self._ensure_user_profiles()

        # Full-stack mode: verify Docker services are reachable
        if self.full_stack:
            self._check_services()

        # Cost tracking accumulators
        self._day_tokens = {'input': 0, 'output': 0}
        self._day_cost = 0.0
        self._total_tokens = {'input': 0, 'output': 0}
        self._total_cost = 0.0
        self._day_costs = []  # list of (day_number, input_tokens, output_tokens, cost)

        # Internal state for each companion (mood, energy, scene)
        self.states = {}
        for cid in companions:
            self.states[cid] = {
                'energy': 0.9,
                'mood': 'neutral',
                'mood_intensity': 0.5,
                'scene': '',
                'mode': 'idle',
                'was_doing': None,
            }
        self._save_all_states()

        logger.info(f"Simulation initialized: companions={companions}, start_day={start_day}")
        logger.info(f"Clock set to: {self.clock.now()}")

    def _load_simulation_config(self):
        """Load all simulation config from external YAML files."""
        import yaml

        # Shared simulation config
        config_path = self.project_root / (self._config_path or 'instances/simulation_config.yaml')
        if config_path.exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
        else:
            logger.warning("No simulation_config.yaml found, using defaults")
            cfg = {}

        self.MOODS = cfg.get('moods', ['neutral', 'relaxed', 'tired', 'creative'])
        self.ACTIVITIES = cfg.get('activities', ['reading', 'coding', 'sketching'])
        self.MODEL_ROTATION = cfg.get('models', ['nvidia/nemotron-3-super-120b-a12b:free'])
        self.OPENER_SEEDS = cfg.get('opener_seeds', ['Share something about your day.'])
        self.TONE_MODIFIERS = cfg.get('tone_modifiers', [''])
        self.LIFE_EVENT_PROB = cfg.get('life_event_probability', 0.6)
        self.LLM_MAX_TOKENS = cfg.get('max_tokens', 350)
        self.LLM_TEMPERATURE = cfg.get('temperature', 0.95)
        self.SUMMARY_MAX_TOKENS = cfg.get('summary_max_tokens', 150)
        self.SUMMARY_TEMPERATURE = cfg.get('summary_temperature', 0.3)

        # Load setting file if specified (e.g., fantasy world context)
        self.setting = {}
        setting_path_str = cfg.get('setting')
        if setting_path_str:
            setting_path = self.project_root / setting_path_str
            if setting_path.exists():
                with open(setting_path) as f:
                    self.setting = yaml.safe_load(f)
                logger.info(f"Loaded setting: {self.setting.get('name', 'unknown')}")

        # Per-companion life events from instances/{cid}/life_events.yaml
        self.LIFE_EVENTS = {}
        for cid in self.companions:
            events_path = self.project_root / f'instances/{cid}/life_events.yaml'
            if events_path.exists():
                with open(events_path) as f:
                    raw = yaml.safe_load(f)
                flat = []
                for category, events in raw.items():
                    for e in events:
                        flat.append((e['event'], e['mood']))
                self.LIFE_EVENTS[cid] = flat
                logger.info(f"Loaded {len(flat)} life events for {cid}")
            else:
                self.LIFE_EVENTS[cid] = []
                logger.warning(f"No life_events.yaml found for {cid}")

    def _update_state_for_time(self, cid: str):
        """Update companion state based on time of day + random life events."""
        hour = self.clock.now().hour
        state = self.states[cid]

        # Energy follows time of day
        if 6 <= hour < 9:
            state['energy'] = round(random.uniform(0.8, 1.0), 2)
        elif 9 <= hour < 15:
            state['energy'] = round(random.uniform(0.6, 0.85), 2)
        elif 15 <= hour < 19:
            state['energy'] = round(random.uniform(0.45, 0.7), 2)
        elif 19 <= hour < 22:
            state['energy'] = round(random.uniform(0.3, 0.55), 2)
        else:
            state['energy'] = round(random.uniform(0.15, 0.35), 2)

        # Life event happens between conversations
        if random.random() < self.LIFE_EVENT_PROB and self.LIFE_EVENTS.get(cid):
            event_text, event_mood = random.choice(self.LIFE_EVENTS[cid])
            state['life_event'] = event_text
            state['mood'] = event_mood
            state['mood_intensity'] = round(random.uniform(0.5, 0.9), 2)
            logger.info(f"  [{cid}] Life event: {event_text[:60]}... -> mood={event_mood}")
        else:
            state['life_event'] = None
            # Mood shifts occasionally even without events
            if random.random() < 0.4:
                state['mood'] = random.choice(self.MOODS)
                state['mood_intensity'] = round(random.uniform(0.3, 0.8), 2)

        # Pick a conversation tone for this conversation
        state['tone'] = random.choice(self.TONE_MODIFIERS)

        # Scene/activity between conversations
        state['was_doing'] = random.choice(self.ACTIVITIES)
        state['mode'] = 'active'

    def _deplete_energy(self, cid: str, minutes: int):
        """Deplete energy during conversation."""
        state = self.states[cid]
        depletion = minutes * 0.002 * random.uniform(0.8, 1.2)
        state['energy'] = max(0.1, round(state['energy'] - depletion, 2))

    def _save_all_states(self):
        """Persist internal states to user_state table."""
        try:
            from src.database.db import get_db
            db = get_db()
            for cid in self.companions:
                state = self.states[cid]
                other = [c for c in self.companions if c != cid][0]
                email = f"{other}@companion.local"
                import json as _json
                db.execute(
                    "UPDATE user_state SET internal_state = %s WHERE email = %s AND companion_id = %s",
                    (_json.dumps(state), email, cid),
                    user_email=email
                )
        except Exception as e:
            logger.warning(f"Failed to save internal states: {e}")

    def _ensure_schema(self):
        """Ensure public and per-user Postgres schemas exist.

        The normal app path bootstraps tables via web_chat -> db, but the
        simulation bypasses that, so we must create them explicitly.
        """
        try:
            from src.database.connection import get_connection
            from src.database.schema_ddl import get_public_schema_ddl
            from src.database.schema_manager import ensure_user_schema

            conn = get_connection()
            try:
                # Create public schema tables (user_profiles, etc.)
                with conn.cursor() as cur:
                    cur.execute(get_public_schema_ddl())
                conn.commit()

                # Create per-companion user schemas
                for cid in self.companions:
                    email = f"{cid}@companion.local"
                    ensure_user_schema(conn, email)
            finally:
                conn.close()

            logger.info("Database schema ensured for simulation")
        except Exception as e:
            logger.error(f"Failed to ensure database schema: {e}")
            raise

    def _ensure_user_profiles(self):
        """Create user_profiles rows for simulation companion emails."""
        try:
            from src.database.db import get_db
            db = get_db()
            for cid in self.companions:
                email = f"{cid}@companion.local"
                db.execute(
                    "INSERT INTO user_profiles (email, display_name) VALUES (%s, %s) ON CONFLICT (email) DO NOTHING",
                    (email, cid.capitalize())
                )
            logger.info("Simulation user profiles ensured")
        except Exception as e:
            logger.error(f"Failed to create user profiles: {e}")

    def _check_services(self):
        """Verify that required Docker services are reachable for full-stack mode."""
        secret = os.environ.get('SIMULATION_API_SECRET')
        if not secret:
            raise RuntimeError(
                "Full-stack mode requires SIMULATION_API_SECRET env var to be set. "
                "This must match the value configured on the API server."
            )
        import requests
        try:
            resp = requests.get(f'{self.api_url}/api/health', timeout=5)
            if resp.status_code == 200:
                logger.info(f"Full-stack mode: API reachable at {self.api_url}")
            else:
                logger.warning(f"Full-stack mode: API returned status {resp.status_code}")
        except requests.ConnectionError:
            raise RuntimeError(
                f"Full-stack mode requires Docker services running. "
                f"Could not connect to {self.api_url}. "
                f"Start services with: docker compose up -d"
            )

    def _generate_message_via_api(self, speaker: str, listener: str, incoming: str) -> str:
        """Generate a message by routing through the HTTP API (full-stack mode).

        The speaker is the companion generating the response; the listener's
        message (incoming) is sent as the 'user' message to the API.
        """
        import requests

        email = f"{listener}@companion.local"
        conv_id = getattr(self, '_current_conversation_id', None)

        payload = {
            'email': email,
            'message': incoming,
            'companion_id': speaker,
        }
        if conv_id is not None:
            payload['conversation_id'] = conv_id

        headers = {
            'X-Simulation-Secret': os.environ.get('SIMULATION_API_SECRET', ''),
        }

        try:
            resp = requests.post(
                f'{self.api_url}/api/chat',
                json=payload,
                headers=headers,
                timeout=120,
            )
            resp.raise_for_status()
            result = resp.json()
            return result.get('response', '').strip()
        except Exception as e:
            logger.error(f"Full-stack API call failed for {speaker}: {e}")
            return f"[generation failed: {e}]"

    def _track_simulation_cost(self, provider, model: str, companion_id: str):
        """Record token usage and cost from the last provider call."""
        usage = provider.get_last_usage()
        if not usage:
            return
        input_tokens = usage.get('input_tokens', 0)
        output_tokens = usage.get('output_tokens', 0)
        if not input_tokens and not output_tokens:
            return

        try:
            from src.services.cost_tracker import get_cost_tracker
            tracker = get_cost_tracker()
            cost = tracker.track_openrouter_call(
                user_id=f"{companion_id}@simulation",
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                model=model,
                call_purpose='simulation',
                companion_id=companion_id,
            )
        except Exception as e:
            logger.debug(f"Cost tracking failed: {e}")
            cost = 0.0

        self._day_tokens['input'] += input_tokens
        self._day_tokens['output'] += output_tokens
        self._day_cost += cost
        self._total_tokens['input'] += input_tokens
        self._total_tokens['output'] += output_tokens
        self._total_cost += cost

    def _reset_day_cost(self):
        """Reset per-day accumulators and record the day's totals."""
        if self._day_tokens['input'] or self._day_tokens['output']:
            self._day_costs.append((
                getattr(self, '_current_day', 0),
                self._day_tokens['input'],
                self._day_tokens['output'],
                self._day_cost,
            ))
        self._day_tokens = {'input': 0, 'output': 0}
        self._day_cost = 0.0

    def run_week(self, week_number: int):
        """Run one week (7 days), then stop for review."""
        start_day = (week_number - 1) * 7
        end_day = start_day + 7

        logger.info(f"\n{'='*60}")
        logger.info(f"WEEK {week_number}: Days {start_day+1} - {end_day}")
        logger.info(f"{'='*60}")

        for day in range(start_day, end_day):
            self._run_day(day)

        # Checkpoint
        self.save_checkpoint(week_number)
        self.print_week_summary(week_number)

    def _run_day(self, day: int):
        """Simulate one day of interaction."""
        self._current_day = day + 1  # 1-indexed for prompts
        day_label = self.clock.now().strftime('%A, %B %d')
        logger.info(f"\n--- Day {day + 1}: {day_label} ---")

        self.emitter.emit('sim:day_start', {'day': day + 1, 'date_label': day_label})
        self.emitter.set_status({
            'status': 'running',
            'day': day + 1,
            'date_label': day_label,
            'clock_time': self.clock.now().isoformat(),
        })

        # Morning schedule generation
        for cid in self.companions:
            self._run_schedule_generation(cid)

        # Reset energy for new day
        for cid in self.companions:
            self.states[cid]['energy'] = round(random.uniform(0.85, 1.0), 2)
            self.states[cid]['mood'] = random.choice(self.MOODS)
            self.states[cid]['mode'] = 'idle'

        # Build conversation pairs for the day based on energy/mood
        # Some days are chatty, some are quiet — like real life
        from itertools import combinations
        all_pairs = list(combinations(self.companions, 2))
        random.shuffle(all_pairs)

        # Base conversation count on average party energy + randomness
        avg_energy = sum(self.states[c].get('energy', 0.5) for c in self.companions) / len(self.companions)

        # Scale by party size: 2 people can have 1-3 chats/day, 4 people need more pairs
        if len(self.companions) <= 2:
            # 2-person: energy drives 2-5 conversations with the same pair
            base = max(2, round(avg_energy * 5))
            num_conversations = max(2, base + random.randint(-1, 1))
            num_conversations = min(num_conversations, 5)
            day_pairs = [all_pairs[0]] * num_conversations  # Same pair, multiple conversations
        else:
            # 3+ people: pick from unique pairs, ensuring coverage
            max_convos = min(len(all_pairs), len(self.companions) + 1)
            energy_bias = max(2, int(avg_energy * max_convos))
            num_conversations = max(2, min(max_convos, energy_bias + random.randint(-1, 1)))
            day_pairs = all_pairs[:num_conversations]

        logger.info(f"  {num_conversations} conversations today (avg energy: {avg_energy:.2f})")

        for conv_idx, (initiator, responder) in enumerate(day_pairs):
            # Randomly swap who initiates
            if random.random() < 0.5:
                initiator, responder = responder, initiator

            hours_gap = random.randint(1, 4)
            self.clock.advance(hours=hours_gap)

            # Update state for this time of day
            for cid in self.companions:
                self._update_state_for_time(cid)
            self._save_all_states()

            logger.info(f"  Conversation {conv_idx+1}/{num_conversations} at {self.clock.now().strftime('%I:%M %p')}")
            self._run_conversation_between(initiator, responder)

            # Deplete energy after conversation
            for cid in self.companions:
                self._deplete_energy(cid, random.randint(10, 30))
                self.states[cid]['mode'] = 'idle'
            self._save_all_states()

        # End of day processing
        tasks_completed = []
        for cid in self.companions:
            completed = self._run_end_of_day(cid)
            tasks_completed.extend(completed)

        self.emitter.emit('sim:day_end', {'day': day + 1, 'tasks_completed': tasks_completed})

        # Weekly tasks (every 7 days)
        if (day + 1) % 7 == 0 and day > 0:
            for cid in self.companions:
                try:
                    self._run_weekly_reflection(cid)
                except Exception as e:
                    logger.warning(f"  [{cid}] weekly_reflection failed: {e}")

        # Monthly tasks (every 30 days)
        if (day + 1) % 30 == 0 and day > 0:
            for cid in self.companions:
                try:
                    self._run_monthly_reflection(cid)
                except Exception as e:
                    logger.warning(f"  [{cid}] monthly_reflection failed: {e}")
                try:
                    self._run_value_inference(cid)
                except Exception as e:
                    logger.warning(f"  [{cid}] value_inference failed: {e}")

        # Log and reset per-day cost accumulators
        if self._day_tokens['input'] or self._day_tokens['output']:
            logger.info(f"  Day {day + 1} tokens: {self._day_tokens['input']} in / {self._day_tokens['output']} out  (${self._day_cost:.4f})")
        self._reset_day_cost()

        # Reset clock to next morning (7 AM) for the following day
        self.clock.set(self.clock.now().replace(hour=7, minute=0, second=0))
        self.clock.advance(days=1)

    _next_conversation_id = 1

    def _new_conversation_id(self):
        """Generate a sequential conversation ID."""
        cid = self._next_conversation_id
        self._next_conversation_id += 1
        return cid

    def _run_conversation_between(self, initiator: str, responder: str):
        """Two specific companions talking to each other."""
        num_exchanges = random.randint(3, 8)
        conv_id = self._new_conversation_id()
        self._current_conversation_id = conv_id
        # All messages in this conversation are stored in the responder's schema,
        # mirroring production where all messages live in the user's schema.
        self._current_conv_email = f"{responder}@companion.local"

        # Rotate model for this conversation
        self._current_model = random.choice(self.MODEL_ROTATION)
        logger.info(f"    {initiator} -> {responder} ({num_exchanges} exchanges) [conv:{conv_id}] [model: {self._current_model.split('/')[-1]}]")
        self.emitter.emit('sim:conversation_start', {
            'initiator': initiator,
            'responder': responder,
            'num_exchanges': num_exchanges,
            'conversation_id': conv_id,
        })

        last_msg = None
        for i in range(num_exchanges):
            try:
                # Pass exchange context for natural wrap-up
                exchange_ctx = {'current': i + 1, 'total': num_exchanges}

                # Initiator sends (or continues) — retry once on empty
                msg = self._generate_message(initiator, responder, incoming=last_msg if i > 0 else None, exchange=exchange_ctx)
                if not msg or msg.startswith('[generation failed') or len(msg.strip()) < 3:
                    import time; time.sleep(5)
                    msg = self._generate_message(initiator, responder, incoming=last_msg if i > 0 else None, exchange=exchange_ctx)
                if not msg or msg.startswith('[generation failed') or len(msg.strip()) < 3:
                    logger.warning(f"    Skipping failed generation for {initiator}")
                    continue
                # Realistic delay: 1-8 minutes + random seconds
                self.clock.advance(minutes=random.randint(1, 8), seconds=random.randint(0, 59))
                self._store_message(initiator, responder, msg, conv_id=conv_id)
                self._emit_message(initiator, responder, msg)

                # Responder replies with realistic typing delay
                reply = self._generate_message(responder, initiator, incoming=msg, exchange=exchange_ctx)
                if not reply or reply.startswith('[generation failed') or len(reply.strip()) < 3:
                    import time; time.sleep(5)
                    reply = self._generate_message(responder, initiator, incoming=msg, exchange=exchange_ctx)
                if not reply or reply.startswith('[generation failed') or len(reply.strip()) < 3:
                    logger.warning(f"    Skipping failed generation for {responder}")
                    last_msg = msg
                    continue
                # Reply delay
                reply_delay = random.randint(1, max(2, min(20, len(msg) // 50)))
                self.clock.advance(minutes=reply_delay, seconds=random.randint(0, 59))
                self._store_message(responder, initiator, reply, conv_id=conv_id)
                self._emit_message(responder, initiator, reply)

                last_msg = reply

            except Exception as e:
                logger.error(f"    Exchange {i+1} failed: {e}")
                continue

        self.emitter.emit('sim:conversation_end', {})

        if self.full_stack:
            # In full-stack mode, Celery tasks are fired by the pipeline automatically.
            # Wait for background tasks to complete before moving on.
            import time
            logger.info(f"    Waiting {self.celery_wait}s for Celery background tasks...")
            time.sleep(self.celery_wait)
        else:
            # Post-conversation processing — extract facts, relationships, curiosity
            # This runs the same logic as Celery tasks but inline
            self._process_conversation(initiator, responder, conv_id, self._current_conv_email)

    def _process_conversation(self, initiator: str, responder: str, conv_id: int, user_email: str = None):
        """Run post-conversation processing inline (normally done by Celery tasks).

        Runs the same 13 background tasks that message_handler.py triggers via
        Celery .delay(), but synchronously so their LLM calls hit the cost tracker.
        """
        try:
            from src.database.db import get_db
            db = get_db()
            conv_email = user_email or f"{responder}@companion.local"
            result = db.execute(
                "SELECT id, sender_name, message_text FROM messages WHERE conversation_id = %s ORDER BY timestamp ASC",
                (conv_id,),
                user_email=conv_email
            )
            rows = result.fetchall()
            if not rows or len(rows) < 2:
                return

            task_count = 0

            # Build paired exchanges (user message + companion response)
            for i in range(0, len(rows) - 1, 2):
                user_msg = rows[i]['message_text']
                user_msg_id = rows[i]['id']
                companion_msg = rows[i + 1]['message_text'] if i + 1 < len(rows) else ''
                companion_msg_id = rows[i + 1]['id'] if i + 1 < len(rows) else None
                if not user_msg or not companion_msg:
                    continue

                speaker_a = rows[i]['sender_name']
                speaker_b = rows[i + 1]['sender_name'] if i + 1 < len(rows) else ''
                email_a = conv_email

                # 1. Fact extraction (LLM call)
                try:
                    from src.tasks.fact_extraction_task import extract_facts_via_llm, store_facts
                    import time; time.sleep(2)  # Rate limit for free tier
                    facts = extract_facts_via_llm(user_msg, companion_msg)
                    if facts:
                        stored = store_facts(facts, email_a)
                        if stored > 0:
                            logger.info(f"    📝 Extracted {stored} facts")
                            task_count += 1
                except Exception as e:
                    logger.debug(f"    Fact extraction failed: {e}")

                # 2. Curiosity extraction (LLM call)
                try:
                    from src.tasks.curiosity_extraction_task import extract_curiosity_from_conversation
                    extract_curiosity_from_conversation(
                        user_message=user_msg,
                        companion_response=companion_msg,
                        user_email=email_a
                    )
                    task_count += 1
                except Exception as e:
                    logger.debug(f"    Curiosity extraction failed: {e}")

                # 3. Curiosity resolution
                try:
                    from src.tasks.curiosity_extraction_task import resolve_discussed_curiosities
                    resolve_discussed_curiosities(
                        companion_response=companion_msg,
                        user_message=user_msg
                    )
                    task_count += 1
                except Exception as e:
                    logger.debug(f"    Curiosity resolution failed: {e}")

                # 4. Relationship extraction (LLM call)
                try:
                    from src.tasks.relationship_extraction_task import extract_relationships
                    time.sleep(2)
                    extract_relationships(email_a, user_msg, companion_msg, user_msg_id)
                    task_count += 1
                except Exception as e:
                    logger.debug(f"    Relationship extraction failed: {e}")

                # 5. Scene extraction (LLM call)
                try:
                    from src.tasks.scene_extraction_task import extract_scene_state
                    time.sleep(2)
                    extract_scene_state(email_a, user_msg, companion_msg, 'simulation')
                    task_count += 1
                except Exception as e:
                    logger.debug(f"    Scene extraction failed: {e}")

                # 6. Internal state update (LLM call)
                try:
                    from src.tasks.internal_state_task import update_internal_state
                    time.sleep(2)
                    update_internal_state(email_a, user_msg, companion_msg)
                    task_count += 1
                except Exception as e:
                    logger.debug(f"    Internal state update failed: {e}")

                # 7. Relationship dynamics (LLM call)
                try:
                    from src.tasks.relationship_dynamics_task import analyze_relationship_dynamics
                    time.sleep(2)
                    analyze_relationship_dynamics(email_a, user_msg, companion_msg)
                    task_count += 1
                except Exception as e:
                    logger.debug(f"    Relationship dynamics failed: {e}")

                # 8. Event synthesis (LLM call)
                try:
                    from src.tasks.event_synthesis_task import detect_and_synthesize_events
                    time.sleep(2)
                    detect_and_synthesize_events(email_a, user_msg, companion_msg, user_msg_id)
                    task_count += 1
                except Exception as e:
                    logger.debug(f"    Event synthesis failed: {e}")

                # 9. Episode tracking (LLM call)
                try:
                    from src.tasks.episode_tracking_task import process_message_episode
                    time.sleep(2)
                    process_message_episode(email_a, user_msg, companion_msg, user_msg_id, companion_msg_id)
                    task_count += 1
                except Exception as e:
                    logger.debug(f"    Episode tracking failed: {e}")

                # 10. Graphiti extraction (knowledge graph)
                try:
                    from src.tasks.graphiti_extraction_task import process_exchange_graphiti
                    time.sleep(2)
                    process_exchange_graphiti(email_a, user_msg, companion_msg)
                    task_count += 1
                except Exception as e:
                    logger.debug(f"    Graphiti extraction failed: {e}")

            if task_count > 0:
                logger.info(f"    ✅ Post-conversation: {task_count} background tasks completed for conv {conv_id}")

        except Exception as e:
            logger.warning(f"    Post-conversation processing failed: {e}")

    def _emit_message(self, speaker: str, listener: str, content: str):
        """Emit a message event to the dashboard."""
        event_data = {
            'speaker': speaker,
            'listener': listener,
            'content': content,
            'timestamp': self.clock.now().isoformat(),
        }

        # Optional analysis (doubles LLM calls)
        if self.with_analysis:
            try:
                from src.core.conversation.message_analyzer import MessageAnalyzer
                analyzer = MessageAnalyzer()
                analysis = analyzer.analyze(content, speaker)
                if analysis:
                    event_data['analysis'] = analysis
            except Exception:
                pass

        self.emitter.emit('sim:message', event_data)

    def _get_all_recent_for_speaker(self, speaker: str, limit: int = 30) -> list:
        """Get recent messages from conversations this speaker participated in,
        scoped to the current simulation's companions only."""
        try:
            from src.database.db import get_db
            db = get_db()
            user_email = getattr(self, '_current_conv_email', f"{speaker}@companion.local")
            # Only pull messages involving companions in THIS simulation
            companion_placeholders = ','.join(['%s'] * len(self.companions))
            result = db.execute(
                f"""SELECT sender_name, message_text, conversation_id FROM messages
                   WHERE (companion_id = %s OR email = %s)
                     AND companion_id IN ({companion_placeholders})
                   ORDER BY timestamp DESC LIMIT %s""",
                (speaker, f"{speaker}@companion.local") + tuple(self.companions) + (limit,),
                user_email=user_email
            )
            rows = result.fetchall()
            mapped = [{'sender': r['sender_name'], 'content': r['message_text'], 'conv_id': r.get('conversation_id')} for r in rows]
            return list(reversed(mapped)) if mapped else []
        except Exception:
            return []

    def _get_day_summary(self, speaker: str, listener: str) -> str:
        """Get a brief summary of ALL recent conversations this speaker has had.
        Gives cross-conversation awareness — knowing what happened with other party members."""
        recent = self._get_all_recent_for_speaker(speaker, limit=15)
        if not recent:
            return ""

        from src.config.persona_config import get_persona_config
        config = get_persona_config(companion_id=speaker)
        other_name = config.primary_user_name

        recent_text = "\n".join([f"{m.get('sender', '???')}: {m['content'][:150]}" for m in recent])

        try:
            import time
            time.sleep(5)  # Rate limit
            from src.llm.openai_provider import OpenAIProvider
            provider = OpenAIProvider(
                api_key=os.getenv('OPENROUTER_API_KEY'),
                model=self.MODEL_ROTATION[0] if self.MODEL_ROTATION else 'nvidia/nemotron-3-super-120b-a12b:free',
                base_url='https://openrouter.ai/api/v1',
                context_limit=131072
            )
            summary_model = self.MODEL_ROTATION[0] if self.MODEL_ROTATION else 'nvidia/nemotron-3-super-120b-a12b:free'
            summary = provider.generate_sync(
                messages=[
                    {"role": "system", "content": "Summarize these recent conversations in 2-3 brief bullet points. These are messages from multiple different conversations with different people. Note who said what and any important information shared. Be concise."},
                    {"role": "user", "content": recent_text}
                ],
                max_tokens=self.SUMMARY_MAX_TOKENS,
                temperature=self.SUMMARY_TEMPERATURE,
            )
            self._track_simulation_cost(provider, summary_model, speaker)
            result = summary.strip() if isinstance(summary, str) else ""
            return result
        except Exception as e:
            logger.debug(f"Summary generation failed: {e}")
            return ""

    def _generate_message(self, speaker: str, listener: str, incoming: str = None, exchange: dict = None) -> str:
        """Generate a message using the companion's full pipeline.

        In full-stack mode with an incoming message, routes through the HTTP API
        so that all background Celery tasks (fact extraction, curiosity, episodes,
        etc.) fire naturally and their LLM costs are tracked.
        """
        # Full-stack mode: route through HTTP API when there's an incoming message
        if self.full_stack and incoming:
            return self._generate_message_via_api(speaker, listener, incoming)

        try:
            from src.config.persona_config import get_persona_config
            config = get_persona_config(companion_id=speaker)

            from src.llm.provider_factory import generate_sync, get_resilient_provider_chain

            # Load personality
            personality_path = self.project_root / f'instances/{speaker}/personality.md'
            personality = ""
            if personality_path.exists():
                personality = personality_path.read_text()

            # Build context
            time_str = self.clock.now().strftime('%A %I:%M %p')
            # Get the listener's display name from their persona config
            try:
                other_config = get_persona_config(companion_id=listener)
                other_name = other_config.companion_short_name
            except Exception:
                other_name = listener.capitalize()
            state = self.states.get(speaker, {})

            # State context for the LLM
            energy_desc = 'high' if state.get('energy', 0.7) > 0.7 else 'moderate' if state.get('energy', 0.5) > 0.4 else 'low'
            state_context = f"Your current state: mood={state.get('mood', 'neutral')}, energy={energy_desc}"
            if state.get('was_doing'):
                state_context += f", you were just {state['was_doing']}"

            # Life event context — something that happened since the last conversation
            life_event_block = ""
            if state.get('life_event'):
                life_event_block = f"\nSomething that just happened to you: {state['life_event']}\nThis is affecting your mood. You might want to talk about it, or it might just color how you respond."

            # Tone modifier
            tone_block = ""
            if state.get('tone'):
                tone_block = f"\nTone for this conversation: {state['tone']}"

            # Load self-profile (includes does_not_have) and the listener's profile
            self_profile_path = self.project_root / f'instances/{speaker}/entity_profiles/{speaker}.yaml'
            self_profile = ""
            if self_profile_path.exists():
                self_profile = self_profile_path.read_text()

            other_profile_path = self.project_root / f'instances/{speaker}/entity_profiles/{listener}.yaml'
            other_profile = ""
            if other_profile_path.exists():
                other_profile = other_profile_path.read_text()

            # Relationship context from persona config
            rel_context = config.relationship_initial_context
            day_num = getattr(self, '_current_day', 1)

            relationship_block = f"""RELATIONSHIP: {rel_context}
You've been texting regularly for about {day_num} day{'s' if day_num != 1 else ''}. You're still getting to know each other — there's comfort but also boundaries. You don't overshare. You don't make big plans casually. Vulnerability comes in small, earned moments, not all at once."""

            boundary = f"""IDENTITY RULES: You are {config.companion_short_name}.
Your profile (what you HAVE and DO NOT HAVE):
{self_profile}

What you know about {other_name} (THEIR life, not yours):
{other_profile}

Do NOT confuse their details with yours. If your profile says "does_not_have: No pets" then you do NOT have pets — period. Never fabricate possessions, pets, vehicles, or relationships not in your profile.

{relationship_block}"""

            if incoming:
                # Mid-conversation: scope to THIS conversation only
                conv_id = getattr(self, '_current_conversation_id', None)
                recent = self._get_recent_messages(speaker, listener, limit=12, conversation_id=conv_id)
                recent_text = "\n".join([f"{m.get('sender', '???')}: {m['content'][:200]}" for m in recent])

                prompt = f"""<character>
{personality}
</character>

<identity>
{boundary}
</identity>

<current_state>
Time: {time_str}
{state_context}
{life_event_block}
</current_state>

<conversation>
{recent_text}

{other_name}: {incoming}
</conversation>
{f"<tone>{state.get('tone')}</tone>" if state.get('tone') else ""}

<task>
Reply to {other_name} as {config.companion_short_name}. This is a text message conversation.
- Write 1-3 sentences max. Be brief and natural.
- Your mood and energy should subtly influence your tone.
- Have opinions. Push back sometimes. Tease them. Don't just agree with everything.
- Do NOT repeat things you already said in this conversation. Read the conversation history above — if you already mentioned a topic, move on or build on it, don't restate it.
- Output ONLY the message text. No narration, no stage directions, no asterisks.
{f"- This conversation is wrapping up. Send a natural closing message — sign off, make a joke, say you gotta go, etc. Do NOT ask a new question or introduce a new topic." if exchange and exchange["current"] >= exchange["total"] else ""}
</task>"""
            else:
                # New conversation: use summary of past conversations, NOT raw messages
                summary = self._get_day_summary(speaker, listener)
                summary_section = f"\n<recent_context>\nWhat you know from recent conversations (with various people, not just {other_name}):\n{summary}\nYou can reference things others told you — e.g. 'Lyra mentioned...' or 'I heard from Kael that...'\n</recent_context>" if summary else ""

                seed = random.choice(self.OPENER_SEEDS)
                prompt = f"""<character>
{personality}
</character>

<identity>
{boundary}
</identity>

<current_state>
Time: {time_str}
{state_context}
{life_event_block}
</current_state>
{summary_section}

<task>
Start a NEW text conversation with {other_name} as {config.companion_short_name}.
Direction: {seed}
{f"Tone: {state.get('tone')}" if state.get('tone') else ""}

Rules:
- Do NOT rehash the same topics from recent conversations. Bring something FRESH.
- If something just happened to you, lead with that.
- Do NOT start with "Hey {other_name}!" — vary your opener. Jump into a thought, question, or observation.
- Write 1-3 sentences max. This is texting.
- Your mood and energy should shape what you say and how you say it.
- Output ONLY the message text. No narration, no stage directions, no asterisks.
</task>"""

            # Rate limit: ~5s between calls to stay under free tier limits
            import time
            time.sleep(5)

            # Use the rotated model for this conversation
            from src.llm.openai_provider import OpenAIProvider
            model = getattr(self, '_current_model', self.MODEL_ROTATION[0])
            provider = OpenAIProvider(
                api_key=os.getenv('OPENROUTER_API_KEY'),
                model=model,
                base_url='https://openrouter.ai/api/v1',
                context_limit=131072
            )

            # Build setting context if available
            setting_block = ""
            if self.setting:
                comm_device = self.setting.get('communication_device', 'text messages')
                comm_style = self.setting.get('communication_style', '')
                setting_desc = self.setting.get('description', '')
                setting_block = f"""
<setting>
{setting_desc}
You communicate via {comm_device}. {comm_style}
</setting>"""

            system_prompt = f"""You are {config.companion_short_name}, sending a message to {other_name} via {self.setting.get('communication_device', 'text')}.
{setting_block}
<rules>
- Stay in character at all times. Your personality file defines who you are.
- Output ONLY the message text. No narration, no actions in asterisks, no meta-commentary, no OOC notes.
- 1-3 sentences max. Keep messages short and natural.
- Magical items (sentient weapons, enchanted lutes, etc.) are OBJECTS you own, not separate people. Reference them as possessions, not as party members.
- Everything in <identity> defines what is YOURS vs THEIRS. Do not confuse them.
</rules>"""

            response = provider.generate_sync(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.LLM_MAX_TOKENS,
                temperature=self.LLM_TEMPERATURE,
            )
            self._track_simulation_cost(provider, model, speaker)

            result = response.strip() if isinstance(response, str) else ""
            # Strip leaked character name prefixes like "Thorne:" or "**Grimble:**"
            if result:
                import re
                result = re.sub(r'^\*{0,2}' + re.escape(config.companion_short_name) + r'\*{0,2}\s*[:]\s*', '', result, count=1)
                result = result.strip().strip('"')
            return result

        except Exception as e:
            logger.error(f"Message generation failed for {speaker}: {e}")
            return f"[generation failed: {e}]"

    def _store_message(self, speaker: str, listener: str, content: str, conv_id: int = None, role: str = 'assistant'):
        """Store a message in the database. Speaker is always the sender."""
        try:
            from src.database.db import get_db
            db = get_db()
            user_email = getattr(self, '_current_conv_email', f"{listener}@companion.local")
            db.execute(
                "INSERT INTO messages (email, sender_name, message_text, timestamp, companion_id, source, conversation_id) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (f"{listener}@companion.local", speaker, content, self.clock.now(), speaker, 'simulation', conv_id),
                user_email=user_email
            )
            logger.debug(f"      [{speaker}] {content[:80]}...")
        except Exception as e:
            logger.error(f"Failed to store message: {e}")

    def _get_recent_messages(self, speaker: str, listener: str, limit: int = 6, conversation_id: int = None) -> list:
        """Get recent messages. If conversation_id is set, scopes to that conversation only.
        Otherwise gets messages between these two specific companions."""
        try:
            from src.database.db import get_db
            db = get_db()
            conv_email = getattr(self, '_current_conv_email', f"{listener}@companion.local")
            if conversation_id:
                # Current conversation only
                result = db.execute(
                    """SELECT sender_name, message_text FROM messages
                       WHERE conversation_id = %s
                       ORDER BY timestamp DESC LIMIT %s""",
                    (conversation_id, limit),
                    user_email=conv_email
                )
            else:
                # All messages between this pair
                result = db.execute(
                    """SELECT sender_name, message_text FROM messages
                       WHERE (companion_id = %s AND email = %s)
                          OR (companion_id = %s AND email = %s)
                       ORDER BY timestamp DESC LIMIT %s""",
                    (speaker, f"{listener}@companion.local",
                     listener, f"{speaker}@companion.local",
                     limit),
                    user_email=conv_email
                )
            rows = result.fetchall()
            mapped = [{'sender': r['sender_name'], 'role': 'assistant' if r['sender_name'] == speaker else 'user', 'content': r['message_text']} for r in rows]
            return list(reversed(mapped)) if mapped else []
        except Exception:
            return []

    def _run_schedule_generation(self, cid: str):
        """Generate daily schedule for a companion."""
        try:
            from src.scheduling.calendar_schedule_generator import generate_daily_schedule
            generate_daily_schedule(target_date=self.clock.now())
            logger.info(f"  [{cid}] Schedule generated")
        except Exception as e:
            logger.warning(f"  [{cid}] Schedule generation skipped: {e}")

    def _run_end_of_day(self, cid: str) -> list:
        """Run all end-of-day autonomous tasks (each fails independently).
        Returns list of completed task names for dashboard reporting."""
        tasks = [
            ('episode_extraction', self._run_episode_extraction),
            ('opinion_formation', self._run_opinion_formation),
            ('curiosity_decay', self._run_curiosity_decay),
            ('confidence_decay', self._run_confidence_decay),
        ]
        completed = []
        for name, func in tasks:
            try:
                func(cid)
                logger.debug(f"  [{cid}] {name} complete")
                completed.append(name)
                self.emitter.emit('sim:task_complete', {
                    'task_name': name,
                    'companion_id': cid,
                    'summary': 'completed',
                })
            except Exception as e:
                logger.warning(f"  [{cid}] {name} failed: {e}")
        return completed

    def _run_episode_extraction(self, cid: str):
        """Extract episodes from today's conversations."""
        from src.tasks.episode_learning_task import run_episode_learning
        run_episode_learning()

    def _run_opinion_formation(self, cid: str):
        """Form/update opinions based on interactions.
        Note: This is a Celery task — skipped when running outside Docker."""
        raise NotImplementedError("Requires Celery worker (run inside Docker)")

    def _run_curiosity_decay(self, cid: str):
        """Decay curiosity urgency for unresolved threads.
        Note: This is a Celery task — skipped when running outside Docker."""
        raise NotImplementedError("Requires Celery worker (run inside Docker)")

    def _run_confidence_decay(self, cid: str):
        """Apply confidence decay to stored facts.
        Note: No standalone decay function available."""
        raise NotImplementedError("No standalone confidence decay function")

    def _run_weekly_reflection(self, cid: str):
        """Run weekly reflection task."""
        logger.info(f"  [{cid}] Running weekly reflection")
        raise NotImplementedError("Requires Celery worker (run inside Docker)")

    def _run_monthly_reflection(self, cid: str):
        """Run monthly reflection task."""
        logger.info(f"  [{cid}] Running monthly reflection")
        raise NotImplementedError("Requires Celery worker (run inside Docker)")

    def _run_value_inference(self, cid: str):
        """Run value inference task."""
        logger.info(f"  [{cid}] Running value inference")
        try:
            from src.tasks.value_inference_task import run_value_inference
            run_value_inference(companion_id=cid)
        except Exception as e:
            logger.debug(f"  [{cid}] Value inference skipped: {e}")

    def save_checkpoint(self, week_number: int):
        """Snapshot database state for review/rollback."""
        timestamp = self.clock.now().strftime('%Y%m%d')
        checkpoint_name = f"week{week_number}_{timestamp}"

        # PostgreSQL dump
        pg_host = os.getenv('POSTGRES_HOST', 'localhost')
        pg_port = os.getenv('POSTGRES_PORT', '5442')
        pg_db = os.getenv('POSTGRES_DB', 'companion')
        pg_user = os.getenv('POSTGRES_USER', 'companion')

        pg_file = self.checkpoint_dir / f"{checkpoint_name}.sql"
        try:
            subprocess.run(
                ['pg_dump', '-h', pg_host, '-p', pg_port, '-U', pg_user, '-d', pg_db, '-f', str(pg_file)],
                env={**os.environ, 'PGPASSWORD': os.getenv('POSTGRES_PASSWORD', '')},
                check=True, capture_output=True
            )
            logger.info(f"Checkpoint saved: {pg_file}")
        except Exception as e:
            logger.warning(f"PostgreSQL checkpoint failed: {e}")

        # Save simulation state
        state_file = self.checkpoint_dir / f"{checkpoint_name}_state.json"
        state = {
            'week': week_number,
            'clock_time': self.clock.now().isoformat(),
            'companions': self.companions,
        }
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Checkpoint saved: week {week_number}")

    def print_week_summary(self, week_number: int):
        """Show what happened this week for human review."""
        print(f"\n{'='*60}")
        print(f"WEEK {week_number} SUMMARY")
        print(f"{'='*60}")

        for cid in self.companions:
            print(f"\n--- {cid.upper()} ---")
            # Use the other companion's email to route to the schema where
            # this companion's data lives (companion cid serves user "other")
            other = [c for c in self.companions if c != cid][0]
            summary_email = f"{other}@companion.local"

            # Count messages
            try:
                from src.database.db import get_db
                db = get_db()
                result = db.execute(
                    "SELECT COUNT(*) as cnt FROM messages WHERE companion_id = %s",
                    (cid,),
                    user_email=summary_email
                )
                row = result.fetchone()
                print(f"  Total messages: {row['cnt'] if row else 0}")
            except Exception:
                print(f"  Total messages: (unavailable)")

            # List opinions
            try:
                result = db.execute(
                    "SELECT topic, opinion_text FROM opinions WHERE companion_id = %s ORDER BY updated_at DESC LIMIT 5",
                    (cid,),
                    user_email=summary_email
                )
                opinions = result.fetchall()
                if opinions:
                    print(f"  Recent opinions:")
                    for op in opinions:
                        print(f"    - {op.get('topic', '?')}: {op.get('opinion_text', '?')[:80]}")
                else:
                    print(f"  Opinions: none yet")
            except Exception:
                print(f"  Opinions: (unavailable)")

            # List curiosity threads
            try:
                result = db.execute(
                    "SELECT topic, urgency FROM curiosity_threads WHERE companion_id = %s AND resolved = false ORDER BY urgency DESC LIMIT 5",
                    (cid,),
                    user_email=summary_email
                )
                threads = result.fetchall()
                if threads:
                    print(f"  Active curiosities:")
                    for t in threads:
                        print(f"    - {t.get('topic', '?')} (urgency: {t.get('urgency', '?')})")
                else:
                    print(f"  Curiosities: none yet")
            except Exception:
                print(f"  Curiosities: (unavailable)")

        # Token and cost summary
        print(f"\n--- COST SUMMARY ---")
        if self._day_costs:
            print(f"  Per-day breakdown:")
            for day_num, t_in, t_out, cost in self._day_costs:
                print(f"    Day {day_num}: {t_in} in / {t_out} out  (${cost:.4f})")
        total_tokens = self._total_tokens['input'] + self._total_tokens['output']
        print(f"  Total tokens: {total_tokens} ({self._total_tokens['input']} in / {self._total_tokens['output']} out)")
        print(f"  Estimated cost: ${self._total_cost:.4f}")

        # Full-stack mode: show cost breakdown by call_purpose from the cost tracker
        if self.full_stack:
            try:
                from src.services.cost_tracker import get_cost_tracker
                tracker = get_cost_tracker()
                for cid in self.companions:
                    breakdown = tracker.get_cost_breakdown_by_purpose(
                        user_id=f"{cid}@companion.local",
                        companion_id=cid,
                    )
                    if breakdown:
                        print(f"\n  Cost by purpose ({cid}):")
                        for entry in sorted(breakdown, key=lambda e: e['total_cost'], reverse=True):
                            print(f"    {entry['call_purpose']}: ${entry['total_cost']:.4f} ({entry['call_count']} calls)")
            except Exception as e:
                logger.debug(f"Could not fetch cost breakdown by purpose: {e}")

        print(f"\n{'='*60}")
        print(f"Review complete. Run --week {week_number + 1} to continue.")
        print(f"Run --rollback week{week_number} to redo this week.")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Checkpoint rollback — restore database to a previous state
# ---------------------------------------------------------------------------

def rollback_checkpoint(checkpoint_name: str, project_root: Path):
    """Restore database from a PostgreSQL dump checkpoint.

    WARNING: This drops and recreates the entire database.
    """
    checkpoint_dir = project_root / 'checkpoints'
    pg_file = checkpoint_dir / f"{checkpoint_name}.sql"

    if not pg_file.exists():
        # Try without timestamp
        matches = list(checkpoint_dir.glob(f"{checkpoint_name}*.sql"))
        if matches:
            pg_file = matches[0]
        else:
            print(f"No checkpoint found matching: {checkpoint_name}")
            return

    pg_host = os.getenv('POSTGRES_HOST', 'localhost')
    pg_port = os.getenv('POSTGRES_PORT', '5442')
    pg_db = os.getenv('POSTGRES_DB', 'companion')
    pg_user = os.getenv('POSTGRES_USER', 'companion')
    pg_pass = os.getenv('POSTGRES_PASSWORD', '')
    if not pg_pass:
        print("ERROR: POSTGRES_PASSWORD environment variable is required")
        return

    # Validate database name to prevent injection (only alphanumeric + underscore)
    import re
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', pg_db):
        print(f"ERROR: Invalid database name: {pg_db}")
        return

    print(f"Restoring checkpoint: {pg_file}")
    print(f"WARNING: This will replace the current database contents.")

    try:
        # Drop and recreate database — pg_db validated above
        env = {**os.environ, 'PGPASSWORD': pg_pass}
        subprocess.run(
            ['psql', '-h', pg_host, '-p', pg_port, '-U', pg_user, '-d', 'postgres',
             '-c', f'DROP DATABASE IF EXISTS {pg_db}'],
            env=env, check=True, capture_output=True
        )
        subprocess.run(
            ['psql', '-h', pg_host, '-p', pg_port, '-U', pg_user, '-d', 'postgres',
             '-c', f'CREATE DATABASE {pg_db}'],
            env=env, check=True, capture_output=True
        )
        subprocess.run(
            ['psql', '-h', pg_host, '-p', pg_port, '-U', pg_user, '-d', pg_db, '-f', str(pg_file)],
            env=env, check=True, capture_output=True
        )
        print(f"Checkpoint restored: {pg_file.name}")
    except Exception as e:
        print(f"Rollback failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Companion Framework — Relationship Simulation')
    parser.add_argument('--week', type=int, help='Run simulation for this week number')
    parser.add_argument('--rollback', type=str, help='Rollback to a checkpoint (e.g., "week1")')
    parser.add_argument('--summary', action='store_true', help='Print current state summary')
    parser.add_argument('--companions', nargs='+', required=True,
                        help='Companion IDs to simulate (e.g. --companions kai mira)')
    parser.add_argument('--config', type=str, default='instances/simulation_config.yaml',
                        help='Path to simulation config YAML (default: instances/simulation_config.yaml)')
    parser.add_argument('--with-analysis', action='store_true',
                        help='Run MessageAnalyzer after each message (doubles LLM calls)')
    parser.add_argument('--full-stack', action='store_true',
                        help='Route messages through the HTTP API for realistic cost/usage data')
    parser.add_argument('--api-url', type=str, default='http://localhost:5001',
                        help='Base URL for the HTTP API (default: http://localhost:5001)')
    parser.add_argument('--celery-wait', type=int, default=5,
                        help='Seconds to wait after each conversation for Celery tasks (default: 5)')
    args = parser.parse_args()

    project_root = find_project_root()
    os.chdir(project_root)
    sys.path.insert(0, str(project_root))

    # Ensure logs dir exists
    (project_root / 'logs').mkdir(exist_ok=True)

    if args.rollback:
        rollback_checkpoint(args.rollback, project_root)
    elif args.week:
        start_day = (args.week - 1) * 7
        runner = SimulationRunner(
            companions=args.companions, start_day=start_day,
            with_analysis=args.with_analysis, config_path=args.config,
            full_stack=args.full_stack, api_url=args.api_url,
            celery_wait=args.celery_wait,
        )
        runner.run_week(args.week)
    elif args.summary:
        runner = SimulationRunner(companions=args.companions, config_path=args.config)
        runner.print_week_summary(0)
    else:
        parser.error("Specify --week N, --rollback NAME, or --summary")


if __name__ == '__main__':
    main()
