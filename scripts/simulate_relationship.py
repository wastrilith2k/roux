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

    def __init__(self, companions: list, start_day: int = 0, with_analysis: bool = False):
        """
        Args:
            companions: list of companion_ids, e.g. ["kai", "mira"]
            start_day: resume from this day (0 = beginning)
            with_analysis: run MessageAnalyzer after each message (doubles LLM calls)
        """
        from src.core.clock import SimulationClock, set_clock
        self.project_root = find_project_root()
        sys.path.insert(0, str(self.project_root))

        self.clock = SimulationClock(start=datetime(2026, 1, 1, 7, 0, tzinfo=PST))
        set_clock(self.clock)

        self.companions = companions
        self.start_day = start_day
        self.with_analysis = with_analysis
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

        logger.info(f"Simulation initialized: companions={companions}, start_day={start_day}")
        logger.info(f"Clock set to: {self.clock.now()}")

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

        # 2-3 conversations throughout the day
        num_conversations = random.randint(2, 3)
        for conv_idx in range(num_conversations):
            hours_gap = random.randint(2, 5)
            self.clock.advance(hours=hours_gap)
            logger.info(f"  Conversation {conv_idx+1}/{num_conversations} at {self.clock.now().strftime('%I:%M %p')}")
            self._run_conversation()

        # End of day processing
        tasks_completed = []
        for cid in self.companions:
            completed = self._run_end_of_day(cid)
            tasks_completed.extend(completed)

        self.emitter.emit('sim:day_end', {'day': day + 1, 'tasks_completed': tasks_completed})

        # Weekly tasks (every 7 days)
        if (day + 1) % 7 == 0 and day > 0:
            for cid in self.companions:
                self._run_weekly_reflection(cid)

        # Monthly tasks (every 30 days)
        if (day + 1) % 30 == 0 and day > 0:
            for cid in self.companions:
                self._run_monthly_reflection(cid)
                self._run_value_inference(cid)

        # Reset clock to next morning (7 AM) for the following day
        self.clock.set(self.clock.now().replace(hour=7, minute=0, second=0))
        self.clock.advance(days=1)

    def _run_conversation(self):
        """Two companions talking to each other (3-8 exchanges)."""
        # Pick who initiates (could use reach-out pressure later)
        initiator, responder = random.sample(self.companions, 2)
        num_exchanges = random.randint(3, 8)

        logger.info(f"    {initiator} -> {responder} ({num_exchanges} exchanges)")
        self.emitter.emit('sim:conversation_start', {
            'initiator': initiator,
            'responder': responder,
            'num_exchanges': num_exchanges,
        })

        for i in range(num_exchanges):
            try:
                # Initiator sends
                msg = self._generate_message(initiator, responder)
                self._store_message(initiator, responder, msg, role='user')
                self._emit_message(initiator, responder, msg)

                # Responder replies
                reply = self._generate_message(responder, initiator, incoming=msg)
                self._store_message(responder, initiator, reply, role='assistant')
                self._emit_message(responder, initiator, reply)

                self.clock.advance(minutes=random.randint(1, 15))

            except Exception as e:
                logger.error(f"    Exchange {i+1} failed: {e}")
                continue

        self.emitter.emit('sim:conversation_end', {})

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

    def _generate_message(self, speaker: str, listener: str, incoming: str = None) -> str:
        """Generate a message using the companion's full pipeline."""
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
            recent = self._get_recent_messages(speaker, listener, limit=6)
            recent_text = "\n".join([f"{'me' if m['role']=='assistant' else listener}: {m['content']}" for m in recent])

            if incoming:
                prompt = f"""You are {config.companion_short_name}. It's {time_str}.

{personality}

Recent conversation with {config.primary_user_name}:
{recent_text}

{config.primary_user_name} just said: {incoming}

Respond naturally, in character. Keep it conversational — this is texting, not an essay."""
            else:
                prompt = f"""You are {config.companion_short_name}. It's {time_str}.

{personality}

Recent conversation with {config.primary_user_name}:
{recent_text}

Start a new conversation with {config.primary_user_name}. Text them something natural — could be about your day, a thought you had, a question, anything. Keep it casual."""

            chain = get_resilient_provider_chain()
            response = generate_sync(
                messages=[
                    {"role": "system", "content": f"You are {config.companion_short_name}. Stay in character. Be brief and natural — this is texting."},
                    {"role": "user", "content": prompt}
                ],
                provider_chain=chain,
                max_tokens=300,
                temperature=0.9,
            )

            return response.strip()

        except Exception as e:
            logger.error(f"Message generation failed for {speaker}: {e}")
            return f"[generation failed: {e}]"

    def _store_message(self, speaker: str, listener: str, content: str, role: str = 'assistant'):
        """Store a message in the database."""
        try:
            from src.database.db import get_db
            db = get_db()
            db.execute(
                "INSERT INTO messages (email, role, content, created_at, companion_id) VALUES (%s, %s, %s, %s, %s)",
                (f"{listener}@companion.local", role, content, self.clock.now(), speaker)
            )
            logger.debug(f"      [{speaker}] {content[:80]}...")
        except Exception as e:
            logger.error(f"Failed to store message: {e}")

    def _get_recent_messages(self, speaker: str, listener: str, limit: int = 6) -> list:
        """Get recent messages between two companions."""
        try:
            from src.database.db import get_db
            db = get_db()
            result = db.execute(
                """SELECT role, content FROM messages
                   WHERE companion_id = %s
                   ORDER BY created_at DESC LIMIT %s""",
                (speaker, limit)
            )
            rows = result.fetchall()
            return list(reversed(rows)) if rows else []
        except Exception:
            return []

    def _run_schedule_generation(self, cid: str):
        """Generate daily schedule for a companion."""
        try:
            from src.scheduling.calendar_schedule_generator import generate_daily_schedule
            generate_daily_schedule(companion_id=cid)
            logger.info(f"  [{cid}] Schedule generated")
        except Exception as e:
            logger.debug(f"  [{cid}] Schedule generation skipped: {e}")

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
                logger.debug(f"  [{cid}] {name} skipped: {e}")
        return completed

    def _run_episode_extraction(self, cid: str):
        """Extract episodes from today's conversations."""
        try:
            from src.tasks.episode_learning_task import run_episode_learning
            run_episode_learning(email=f"{cid}@companion.local", companion_id=cid)
        except Exception:
            pass

    def _run_opinion_formation(self, cid: str):
        """Form/update opinions based on interactions."""
        try:
            from src.tasks.opinion_formation_task import run_opinion_formation
            run_opinion_formation(companion_id=cid)
        except Exception:
            pass

    def _run_curiosity_decay(self, cid: str):
        """Decay curiosity urgency for unresolved threads."""
        try:
            from src.tasks.curiosity_extraction_task import decay_curiosity_urgency
            decay_curiosity_urgency(companion_id=cid)
        except Exception:
            pass

    def _run_confidence_decay(self, cid: str):
        """Apply confidence decay to stored facts."""
        try:
            from src.memory.confidence_decay import apply_confidence_decay
            apply_confidence_decay(companion_id=cid)
        except Exception:
            pass

    def _run_weekly_reflection(self, cid: str):
        """Run weekly reflection task."""
        logger.info(f"  [{cid}] Running weekly reflection")
        try:
            from src.tasks.weekly_reflection_task import run_weekly_reflection
            run_weekly_reflection(companion_id=cid)
        except Exception as e:
            logger.debug(f"  [{cid}] Weekly reflection skipped: {e}")

    def _run_monthly_reflection(self, cid: str):
        """Run monthly reflection task."""
        logger.info(f"  [{cid}] Running monthly reflection")
        try:
            from src.tasks.monthly_reflection_task import run_monthly_reflection
            run_monthly_reflection(companion_id=cid)
        except Exception as e:
            logger.debug(f"  [{cid}] Monthly reflection skipped: {e}")

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

            # Count messages
            try:
                from src.database.db import get_db
                db = get_db()
                result = db.execute(
                    "SELECT COUNT(*) as cnt FROM messages WHERE companion_id = %s",
                    (cid,)
                )
                row = result.fetchone()
                print(f"  Total messages: {row['cnt'] if row else 0}")
            except Exception:
                print(f"  Total messages: (unavailable)")

            # List opinions
            try:
                result = db.execute(
                    "SELECT topic, opinion_text FROM opinions WHERE companion_id = %s ORDER BY updated_at DESC LIMIT 5",
                    (cid,)
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
                    (cid,)
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
    parser.add_argument('--companions', nargs='+', default=['kai', 'mira'],
                        help='Companion IDs to simulate (default: kai mira)')
    parser.add_argument('--with-analysis', action='store_true',
                        help='Run MessageAnalyzer after each message (doubles LLM calls)')
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
        runner = SimulationRunner(companions=args.companions, start_day=start_day, with_analysis=args.with_analysis)
        runner.run_week(args.week)
    elif args.summary:
        runner = SimulationRunner(companions=args.companions)
        runner.print_week_summary(0)
    else:
        parser.error("Specify --week N, --rollback NAME, or --summary")


if __name__ == '__main__':
    main()
