"""
Celery Application -- async task processing backbone.

WHAT: Configures the Celery app with Redis broker, 30+ task includes across
      all subsystems, and an extensive beat_schedule for periodic jobs (fact
      extraction, memory cleanup, proactive messaging, schedule generation,
      cost tracking, health checks, etc.).

WHY:  Many companion operations are too slow for the request/response cycle:
      image generation (60-120s), fact extraction, memory consolidation,
      proactive message planning. Celery handles these asynchronously so the
      WebSocket response stays fast. Beat scheduling replaces cron for
      periodic maintenance.

HOW:  Redis as broker and result backend. Tasks are auto-discovered from
      `src/tasks/` modules listed in `include`. Beat schedule uses crontab
      triggers for most jobs. Sentry integration (optional) catches worker
      errors. Task safety settings (acks_late, reject_on_worker_lost) prevent
      message loss during deploys.
"""
from celery import Celery
import os

# Sentry is optional - only load if installed
try:
    import sentry_sdk
    from sentry_sdk.integrations.celery import CeleryIntegration
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False

# Initialize Sentry for Celery worker error tracking
sentry_dsn = os.getenv('SENTRY_DSN')
if sentry_dsn and SENTRY_AVAILABLE:
    sentry_sdk.init(
        dsn=sentry_dsn,
        integrations=[CeleryIntegration()],
        traces_sample_rate=0.0,  # Disabled — free tier has limited quota
        environment=os.getenv('FLASK_ENV', 'production'),
        release=os.getenv('GIT_COMMIT', 'unknown'),
    )
    print(f"✅ Sentry initialized for Celery worker")
elif sentry_dsn and not SENTRY_AVAILABLE:
    print("⚠️  SENTRY_DSN set but sentry_sdk not installed - skipping Sentry init")

# Initialize Celery
celery_app = Celery(
    'companion_tasks',
    broker=os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    backend=os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'),
)

# Configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='America/Los_Angeles',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes max per task
    task_soft_time_limit=540,  # 9 minutes soft limit (warning)
    worker_prefetch_multiplier=1,  # Take one task at a time
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks (prevent memory leaks)
    broker_connection_retry_on_startup=True,  # Retry connecting to Redis on startup
    broker_connection_retry=True,  # Retry on connection loss (not just startup)
    broker_connection_max_retries=10,  # Retry up to 10 times
    broker_heartbeat=30,  # Send heartbeat every 30s to detect dead connections
    task_acks_late=True,  # Acknowledge tasks after completion (safer if worker crashes)
    worker_disable_rate_limits=False,  # Allow rate limiting
    task_reject_on_worker_lost=True,  # Requeue tasks if worker dies
)

# Explicitly include tasks - autodiscover wasn't finding them
celery_app.conf.update(
    include=[
        'src.tasks.episodic_embedding_task',  # Embed new messages for semantic search
        'src.tasks.correction_task',  # Learn from user corrections
        'src.tasks.scene_extraction_task',  # Scene state tracking
        'src.tasks.internal_state_task',  # Companion's internal state (energy, needs)
        'src.tasks.relationship_dynamics_task',  # Gottman-informed relationship tracking
        'src.tasks.fact_extraction_task',  # Extract facts from conversations
        'src.tasks.relationship_extraction_task',  # Extract relationships from conversations
        'src.tasks.graphiti_extraction_task',  # Temporal knowledge graph extraction
        'src.tasks.event_synthesis_task',  # Event-based narrative synthesis
        'src.tasks.episode_tracking_task',  # Conversation episode tracking
        'src.tasks.image_generation_task',  # Async image generation via RunComfy
        'src.tasks.core_memory_task',  # Core memory refresh (weekly)
        'src.tasks.daily_summary_task',  # Daily conversation summaries
        'src.tasks.biography_refresh_task',  # Synthesized biography refresh (daily)
        'src.tasks.event_consolidation_task',  # Event consolidation (daily)
        'src.tasks.episode_learning_task',  # Episode learning - extract lessons from conversations
        'src.tasks.reflection_task',  # Daily reflection - analyze summaries for insights
        'src.tasks.curiosity_extraction_task',  # Extract curiosity triggers from conversations
        'src.tasks.opinion_formation_task',  # Form opinions from reflections
        'src.tasks.autonomous_action_task',  # Autonomous actions (research, projects, etc.)
        'src.tasks.value_inference_task',  # Value inference - emergent personality from conversation history
        'src.tasks.relationship_evaluation_task',  # Relationship evaluation - companion defines own relationship status
        'src.tasks.goal_planning_task',  # Goal decomposition - break goals into actionable steps
        'src.tasks.memory_pruning_task',  # PostgreSQL-based fact pruning (weekly)
        'src.tasks.memory_gap_analysis_task',  # Memory gap detection → curiosity (daily)
        'src.tasks.entity_profile_fact_validator',  # Validate facts against entity profiles (daily)
        'src.tasks.interaction_outcome_task',  # Per-message interaction outcome tracking
        'src.tasks.weekly_reflection_task',  # Weekly pattern analysis reflection
        'src.tasks.monthly_reflection_task',  # Monthly relationship evolution reflection
        'src.tasks.calendar_schedule_task',  # Daily calendar schedule generation
        'src.tasks.reminder_check_task',  # Due reminders → queued_thoughts
        'src.tasks.gmail_check_task',  # Gmail inbox awareness
        'src.tasks.goal_signal_task',  # Signal-based goal formation
    ]
)

# Celery Beat schedule for periodic tasks
from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    'daily-biography-refresh': {
        'task': 'tasks.refresh_biographies',
        'schedule': 60 * 60 * 24,  # Every 24 hours

    },
    'daily-event-consolidation': {
        'task': 'tasks.consolidate_events',
        'schedule': crontab(hour=3, minute=0),  # 3:00 AM Pacific daily

    },
    'daily-episode-learning': {
        'task': 'tasks.learn_from_episodes',
        'schedule': crontab(hour=4, minute=0),  # 4:00 AM Pacific daily (after consolidation)

    },
    'daily-reflection': {
        'task': 'tasks.reflection_task.reflect_on_day',
        'schedule': crontab(hour=0, minute=30),  # 12:30 AM Pacific daily (after daily summary at 12:05)

    },
    'daily-opinion-formation': {
        'task': 'tasks.opinion_formation.form_opinions',
        'schedule': crontab(hour=1, minute=0),  # 1:00 AM Pacific daily (after reflection at 12:30)

    },
    'weekly-opinion-decay': {
        'task': 'tasks.opinion_formation.decay_weak_opinions',
        'schedule': crontab(hour=2, minute=0, day_of_week=0),  # 2:00 AM Sunday (weekly)

    },
    # Autonomous actions — now event-driven via AlwaysOnService._check_goal_actions()
    # (celery task kept importable for debugging, but no longer scheduled)
    #
    # Value inference - analyze the companion's messages to infer emergent personality
    'weekly-value-inference': {
        'task': 'tasks.value_inference.run_weekly',
        'schedule': crontab(hour=3, minute=30, day_of_week=3),  # 3:30 AM Wednesday (weekly)

    },
    'monthly-value-inference-full': {
        'task': 'tasks.value_inference.run_full_history',
        'schedule': crontab(hour=4, minute=0, day_of_month=1),  # 4:00 AM 1st of month

    },
    # Relationship evaluation - companion decides how to define the relationship
    'weekly-relationship-evaluation': {
        'task': 'tasks.relationship_evaluation.evaluate_weekly',
        'schedule': crontab(hour=2, minute=30, day_of_week=2),  # Tuesday 2:30 AM Pacific (after Monday reflection + opinions)

    },
    # Image orphan recovery - check for stuck image generation requests
    'image-orphan-recovery': {
        'task': 'src.tasks.image_generation_task.recover_orphaned_images',
        'schedule': 60 * 5,  # Every 5 minutes

    },
    # Curiosity urgency update - increase urgency for unexplored topics
    'daily-curiosity-urgency': {
        'task': 'tasks.curiosity_extraction.update_curiosity_urgency',
        'schedule': crontab(hour=5, minute=0),  # 5:00 AM Pacific daily

    },
    # Goal planning - decompose unplanned goals into steps
    'goal-planning-decompose': {
        'task': 'tasks.goal_planning.decompose_unplanned_goals',
        'schedule': crontab(hour='6,12,18,0', minute=30),  # Every 6h: 6:30AM, 12:30PM, 6:30PM, 12:30AM

    },
    # Memory pruning - archive stale low-retention facts
    'weekly-memory-pruning': {
        'task': 'tasks.memory_pruning.prune_stale_facts',
        'schedule': crontab(hour=5, minute=30, day_of_week=0),  # Sunday 5:30 AM Pacific
        'kwargs': {'dry_run': False},

    },
    # Memory gap analysis - identify knowledge gaps and generate curiosity
    'daily-memory-gap-analysis': {
        'task': 'tasks.memory_gap_analysis.analyze_memory_gaps',
        'schedule': crontab(hour=6, minute=0),  # 6:00 AM Pacific daily

    },
    # Entity profile fact validation - archive facts that contradict YAML profiles
    'daily-entity-profile-fact-validation': {
        'task': 'tasks.entity_profile_fact_validator.validate_facts_against_profiles',
        'schedule': crontab(hour=3, minute=30),  # 3:30 AM Pacific daily (after biography refresh)

    },
    # Weekly reflection - analyze patterns across the week
    'weekly-reflection': {
        'task': 'tasks.reflection_task.reflect_on_week',
        'schedule': crontab(hour=3, minute=0, day_of_week=0),  # Sunday 3:00 AM Pacific

    },
    # Monthly reflection - relationship evolution analysis
    'monthly-reflection': {
        'task': 'tasks.reflection_task.reflect_on_month',
        'schedule': crontab(hour=4, minute=0, day_of_month=1),  # 1st of month 4:00 AM Pacific

    },
    # Daily calendar schedule generation - generates the companion's day plan
    'daily-calendar-schedule': {
        'task': 'tasks.calendar_schedule.generate_daily_plan',
        'schedule': crontab(hour=5, minute=30),  # 5:30 AM Pacific daily (before her 6 AM wake-up)

    },
    # Reminder check - surface due reminders to queued_thoughts
    'check-due-reminders': {
        'task': 'tasks.reminder_check.check_due_reminders',
        'schedule': crontab(minute=0, hour='8,10,12,14,16,18,20'),  # Every 2h during waking hours

    },
    # Gmail inbox awareness - surface interesting emails
    'check-gmail-inbox': {
        'task': 'tasks.gmail_check.check_inbox',
        'schedule': crontab(minute='5,35', hour='8,9,10,11,12,13,14,15,16,17,18,19,20,21'),  # Every 30min during waking hours

    },
    # Signal-based goal formation - curiosities, opinions, findings → goals
    'goal-formation-signals': {
        'task': 'tasks.goal_signals.form_from_signals',
        'schedule': crontab(hour='9,21', minute=0),  # 9 AM and 9 PM

    },
    # Daily incremental value refresh - volatile categories only (private_thoughts, feelings, unresolved)
    'daily-value-refresh': {
        'task': 'tasks.value_inference.run_incremental',
        'schedule': crontab(hour=1, minute=0),  # 1 AM daily, after reflection at 12:30 AM

    },
}

# Also export as 'app' for backward compatibility
app = celery_app

# Export both names
__all__ = ['celery_app', 'app']
