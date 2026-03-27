"""
Work Projects — persistent work identity for the companion.

WHAT: Manages 2-4 ongoing work projects that give the companion specific
      things to be working on. Projects have names, progress, feelings,
      and deadlines that evolve over time.

WHY:  Without persistent projects, the companion says "I'm working" but
      has nothing specific to work ON. This gives her real tasks to be
      excited about, frustrated by, or proud of finishing.

HOW:  - Projects are stored in PostgreSQL (companion_work_projects)
      - An LLM generates/evolves projects based on the companion's
        employment info from their entity profile
      - Projects are referenced in schedule generation, behavior context,
        and reach-out messages
      - A daily task advances projects (progress, feelings, completion)

Usage:
    from src.autonomy.work_projects import get_work_project_manager

    manager = get_work_project_manager()
    projects = manager.get_active_projects()
    context = manager.format_for_prompt()
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from src.config.models import FIREWORKS_DEFAULT_MODEL as FIREWORKS_MODEL

logger = logging.getLogger(__name__)


def _get_db_connection():
    """Get a psycopg2 connection."""
    import psycopg2
    return psycopg2.connect(
        host=os.environ.get('POSTGRES_HOST', 'postgres'),
        port=os.environ.get('POSTGRES_PORT', '5432'),
        dbname=os.environ.get('POSTGRES_DB', 'companion'),
        user=os.environ.get('POSTGRES_USER', 'companion'),
        password=os.environ.get('POSTGRES_PASSWORD', '')
    )


def _ensure_table(conn) -> None:
    """Create the work projects table if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS companion_work_projects (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                status TEXT DEFAULT 'active',
                progress_pct INTEGER DEFAULT 0,
                feeling TEXT DEFAULT '',
                deadline TEXT DEFAULT '',
                last_worked DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    conn.commit()


class WorkProjectManager:
    """Manages the companion's work projects."""

    def get_active_projects(self) -> List[Dict]:
        """Get all active work projects."""
        conn = _get_db_connection()
        try:
            _ensure_table(conn)
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, name, description, status, progress_pct,
                           feeling, deadline, last_worked
                    FROM companion_work_projects
                    WHERE status IN ('active', 'blocked')
                    ORDER BY updated_at DESC
                """)
                return [
                    {
                        'id': r[0], 'name': r[1], 'description': r[2],
                        'status': r[3], 'progress_pct': r[4],
                        'feeling': r[5], 'deadline': r[6],
                        'last_worked': str(r[7]) if r[7] else None,
                    }
                    for r in cur.fetchall()
                ]
        except Exception as e:
            logger.warning(f"Failed to get active projects: {e}")
            return []
        finally:
            conn.close()

    def get_all_projects(self) -> List[Dict]:
        """Get all projects including completed ones."""
        conn = _get_db_connection()
        try:
            _ensure_table(conn)
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, name, description, status, progress_pct,
                           feeling, deadline, last_worked, created_at
                    FROM companion_work_projects
                    ORDER BY
                        CASE status
                            WHEN 'active' THEN 0
                            WHEN 'blocked' THEN 1
                            WHEN 'completed' THEN 2
                        END,
                        updated_at DESC
                """)
                return [
                    {
                        'id': r[0], 'name': r[1], 'description': r[2],
                        'status': r[3], 'progress_pct': r[4],
                        'feeling': r[5], 'deadline': r[6],
                        'last_worked': str(r[7]) if r[7] else None,
                        'created_at': str(r[8]) if r[8] else None,
                    }
                    for r in cur.fetchall()
                ]
        except Exception as e:
            logger.warning(f"Failed to get all projects: {e}")
            return []
        finally:
            conn.close()

    def generate_projects(self, force: bool = False) -> List[Dict]:
        """
        Generate or evolve work projects using LLM.

        Reads the companion's employment info from their entity profile
        and generates realistic work projects. If projects already exist,
        evolves them (advances progress, changes feelings, completes some,
        starts new ones).
        """
        existing = self.get_active_projects()

        if existing and not force:
            logger.debug(f"Already have {len(existing)} active projects")
            return existing

        # Get companion's employment info from entity profile
        employment = self._get_employment_info()
        if not employment:
            logger.warning("No employment info found in entity profile")
            return existing

        try:
            from openai import OpenAI
            client = OpenAI(
                base_url="https://api.fireworks.ai/inference/v1",
                api_key=os.getenv('FIREWORKS_API_KEY')
            )

            from src.config.persona_config import get_persona_config
            _pc = get_persona_config()

            existing_desc = ""
            if existing:
                existing_desc = "CURRENT PROJECTS:\n"
                for p in existing:
                    existing_desc += (
                        f"- {p['name']} ({p['progress_pct']}% done, "
                        f"feeling: {p['feeling'] or 'neutral'})\n"
                    )
                existing_desc += (
                    "\nEvolve these: advance progress on some, maybe "
                    "complete one if it's near 100%, and optionally start "
                    "a new one if there are fewer than 3.\n"
                )

            prompt = f"""Generate work projects for {_pc.companion_short_name}.

EMPLOYMENT:
- Job: {employment.get('job_title', 'unknown')}
- Employer: {employment.get('employer', 'unknown')}
- Notes: {employment.get('note', '')}

{existing_desc}

Generate 2-4 realistic work projects. Each should feel like something
a real person in this role would be working on. Include:
- Specific project names (not generic like "documentation task")
- A mix of: one exciting/creative project, one routine/deadline project,
  and optionally one that's frustrating or challenging
- Realistic progress percentages
- How they FEEL about each project (excited, bored, stressed, proud, etc.)

/no_think
Return as JSON array only:
[
  {{
    "name": "specific project name",
    "description": "1-2 sentences about what this project involves",
    "progress_pct": 0-100,
    "feeling": "one word or short phrase",
    "deadline": "optional deadline like 'next Friday' or 'end of month' or ''",
    "status": "active"
  }}
]"""

            import re
            response = client.chat.completions.create(
                model=FIREWORKS_MODEL,
                max_tokens=800,
                temperature=0.85,
                messages=[{"role": "user", "content": prompt}]
            )

            content = response.choices[0].message.content.strip()
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            content = re.sub(r'<think>.*', '', content, flags=re.DOTALL)
            content = content.strip()

            if '```' in content:
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()

            projects = json.loads(content)

            if not isinstance(projects, list):
                raise ValueError("Expected JSON array")

            # Store to DB
            self._store_projects(projects)
            logger.info(f"Generated {len(projects)} work projects")
            return self.get_active_projects()

        except Exception as e:
            logger.error(f"Failed to generate work projects: {e}")
            return existing

    def advance_projects(self) -> Dict:
        """
        Daily evolution of projects via LLM.

        Called by celery-beat daily. Advances progress, changes feelings,
        completes projects near 100%, and starts new ones if needed.
        """
        existing = self.get_active_projects()
        if not existing:
            return self.generate_projects()

        employment = self._get_employment_info()

        try:
            from openai import OpenAI
            import re
            client = OpenAI(
                base_url="https://api.fireworks.ai/inference/v1",
                api_key=os.getenv('FIREWORKS_API_KEY')
            )

            from src.config.persona_config import get_persona_config
            _pc = get_persona_config()

            projects_desc = "\n".join([
                f"- ID {p['id']}: {p['name']} ({p['progress_pct']}%, "
                f"feeling: {p['feeling'] or 'neutral'}, "
                f"deadline: {p['deadline'] or 'none'})"
                for p in existing
            ])

            prompt = f"""Advance {_pc.companion_short_name}'s work projects by one day.

JOB: {employment.get('job_title', '')} at {employment.get('employer', '')}

CURRENT PROJECTS:
{projects_desc}

For each project, decide:
1. Advance progress by 5-20% (realistic daily progress)
2. Update feeling (may change day to day)
3. If progress >= 90%, maybe complete it (status: "completed")
4. If fewer than 2 active projects remain, add a new one

/no_think
Return JSON array with ALL projects (updated + any new ones):
[
  {{
    "id": existing_id_or_null,
    "name": "project name",
    "description": "updated description",
    "progress_pct": updated_number,
    "feeling": "current feeling",
    "deadline": "deadline or empty",
    "status": "active or completed"
  }}
]"""

            response = client.chat.completions.create(
                model=FIREWORKS_MODEL,
                max_tokens=800,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )

            content = response.choices[0].message.content.strip()
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            content = re.sub(r'<think>.*', '', content, flags=re.DOTALL)
            content = content.strip()
            if '```' in content:
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()

            updates = json.loads(content)
            self._apply_updates(updates)

            logger.info(f"Advanced {len(updates)} work projects")
            return {'updated': len(updates)}

        except Exception as e:
            logger.error(f"Failed to advance projects: {e}")
            return {'error': str(e)}

    def format_for_prompt(self) -> str:
        """Format active projects for injection into the system prompt."""
        projects = self.get_active_projects()
        if not projects:
            return ""

        lines = ["[YOUR WORK PROJECTS - things you're actively working on]"]
        for p in projects:
            line = f"- {p['name']} ({p['progress_pct']}% done"
            if p['feeling']:
                line += f", feeling: {p['feeling']}"
            if p['deadline']:
                line += f", due: {p['deadline']}"
            line += f"): {p['description']}"
            lines.append(line)

        lines.append(
            "(These are YOUR real work projects. You can reference them "
            "when talking about your day, complain about deadlines, "
            "share excitement about progress, etc.)"
        )
        return "\n".join(lines)

    def format_for_schedule(self) -> List[Dict]:
        """Format projects for the schedule generator skeleton."""
        projects = self.get_active_projects()
        return [
            {'name': p['name'], 'description': p['description'][:100]}
            for p in projects
        ]

    def _get_employment_info(self) -> Dict:
        """Read companion's employment info from entity profile."""
        try:
            from src.core.entity_profile_loader import get_entity_profile_loader
            from src.config.persona_config import get_persona_config

            _pc = get_persona_config()
            loader = get_entity_profile_loader()
            profile = loader.get_profile(_pc.companion_entity_profile)

            if profile and 'employment' in profile:
                return profile['employment']
            return {}
        except Exception as e:
            logger.warning(f"Could not load employment info: {e}")
            return {}

    def _store_projects(self, projects: List[Dict]) -> None:
        """Store new projects to DB, clearing existing active ones."""
        conn = _get_db_connection()
        try:
            _ensure_table(conn)
            with conn.cursor() as cur:
                # Archive existing active projects
                cur.execute("""
                    UPDATE companion_work_projects
                    SET status = 'archived'
                    WHERE status = 'active'
                """)
                # Insert new projects
                for p in projects:
                    cur.execute("""
                        INSERT INTO companion_work_projects
                            (name, description, status, progress_pct,
                             feeling, deadline, last_worked)
                        VALUES (%s, %s, %s, %s, %s, %s, CURRENT_DATE)
                    """, (
                        p.get('name', 'Untitled'),
                        p.get('description', ''),
                        p.get('status', 'active'),
                        p.get('progress_pct', 0),
                        p.get('feeling', ''),
                        p.get('deadline', ''),
                    ))
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to store projects: {e}")
        finally:
            conn.close()

    def _apply_updates(self, updates: List[Dict]) -> None:
        """Apply daily evolution updates to projects."""
        conn = _get_db_connection()
        try:
            _ensure_table(conn)
            with conn.cursor() as cur:
                for u in updates:
                    if u.get('id'):
                        # Update existing project
                        cur.execute("""
                            UPDATE companion_work_projects
                            SET progress_pct = %s,
                                feeling = %s,
                                status = %s,
                                deadline = %s,
                                description = %s,
                                last_worked = CURRENT_DATE,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE id = %s
                        """, (
                            u.get('progress_pct', 0),
                            u.get('feeling', ''),
                            u.get('status', 'active'),
                            u.get('deadline', ''),
                            u.get('description', ''),
                            u['id'],
                        ))
                    else:
                        # New project
                        cur.execute("""
                            INSERT INTO companion_work_projects
                                (name, description, status, progress_pct,
                                 feeling, deadline, last_worked)
                            VALUES (%s, %s, %s, %s, %s, %s, CURRENT_DATE)
                        """, (
                            u.get('name', 'New Project'),
                            u.get('description', ''),
                            u.get('status', 'active'),
                            u.get('progress_pct', 0),
                            u.get('feeling', ''),
                            u.get('deadline', ''),
                        ))
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to apply project updates: {e}")
        finally:
            conn.close()


# Singleton
_manager: Optional[WorkProjectManager] = None


def get_work_project_manager() -> WorkProjectManager:
    """Get or create the WorkProjectManager singleton."""
    global _manager
    if _manager is None:
        _manager = WorkProjectManager()
    return _manager
