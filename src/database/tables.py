"""
Table name constants for all PostgreSQL tables.

Every SQL query in the codebase should reference these constants instead of
hardcoding table names. This makes it easy to audit the full schema and
share the constants across deployments.

Tables are split into two groups:
- PUBLIC_TABLES: shared across all users (auth, profiles, reference data)
- USER_SCHEMA_TABLES: exist per-user in their own PostgreSQL schema
"""

# -- Public schema (shared) ---------------------------------------------------
USERS = "users"
SESSIONS = "sessions"
USER_PROFILES = "user_profiles"
USER_COMPANIONS = "user_companions"
SERVICE_LINKS = "service_links"

# -- User schema: messages -----------------------------------------------------
MESSAGES = "messages"

# -- User schema: facts & knowledge --------------------------------------------
FACTS = "facts"
PENDING_FACTS = "pending_facts"
FACT_LINKS = "fact_links"
DETECTED_CONTRADICTIONS = "detected_contradictions"
CORRECTIONS = "corrections"

# -- User schema: episodes -----------------------------------------------------
EPISODES = "episodes"
EPISODE_MESSAGES = "episode_messages"
EPISODE_PATTERNS = "episode_patterns"

# -- User schema: observations -------------------------------------------------
OBSERVATIONS = "observations"
OBSERVATION_REFLECTIONS = "observation_reflections"
COMPANION_OBSERVATIONS = "companion_observations"

# -- User schema: memory synthesis ---------------------------------------------
SYNTHESIZED_EVENTS = "synthesized_events"
SYNTHESIZED_PARAGRAPHS = "synthesized_paragraphs"
DAILY_SUMMARIES = "daily_summaries"

# -- User schema: relationships ------------------------------------------------
RELATIONSHIPS = "relationships"

# -- User schema: autonomy -----------------------------------------------------
COMPANION_GOALS = "companion_goals"
COMPANION_GOAL_STEPS = "companion_goal_steps"
COMPANION_AUTONOMOUS_TASKS = "companion_autonomous_tasks"
COMPANION_OPINIONS = "companion_opinions"
COMPANION_JOURNAL = "companion_journal"
COMPANION_REMINDERS = "companion_reminders"
CURIOSITY_THREADS = "curiosity_threads"
INTERACTION_OUTCOMES = "interaction_outcomes"

# -- User schema: hidden/private -----------------------------------------------
INTERNAL_STATE_VALUES = "_companion_internal_state_v"
RELATIONSHIP_EVAL = "_companion_relationship_eval"

# -- User schema: scene & conversation state -----------------------------------
SCENE_STATE = "scene_state"
STATE = "state"
CONVERSATION_PATTERNS = "conversation_patterns"
CONVERSATION_PATTERNS_STATS = "conversation_patterns_stats"
EMOTION_ANALYTICS = "emotion_analytics"

# -- User schema: user preferences ---------------------------------------------
USER_STATE = "user_state"
USER_PREFERENCES = "user_preferences"

# -- User schema: feed ---------------------------------------------------------
FEED_POSTS = "feed_posts"

# -- User schema: scheduling & events ------------------------------------------
UPCOMING_EVENTS = "upcoming_events"

# -- User schema: benchmarks ---------------------------------------------------
BENCHMARK_QUESTIONS = "benchmark_questions"
BENCHMARK_RESULTS = "benchmark_results"
BENCHMARK_RUNS = "benchmark_runs"

# -- User schema: images -------------------------------------------------------
IMAGE_GENERATION_REQUESTS = "image_generation_requests"

# -- User schema: cost tracking ------------------------------------------------
DAILY_COST_SUMMARY = "daily_cost_summary"
BUDGET_LIMITS = "budget_limits"
OPENROUTER_USAGE = "openrouter_usage"
FIREWORKS_USAGE = "fireworks_usage"
OPENAI_USAGE = "openai_usage"
GOOGLE_API_USAGE = "google_api_usage"
HEDRA_USAGE = "hedra_usage"
RUNCOMFY_USAGE = "runcomfy_usage"
TWILIO_USAGE = "twilio_usage"

# -- User schema: error tracking -----------------------------------------------
ERROR_LOG = "error_log"

# -- Lookup sets ---------------------------------------------------------------
PUBLIC_TABLES = {USERS, SESSIONS, USER_PROFILES, USER_COMPANIONS, SERVICE_LINKS}

USER_SCHEMA_TABLES = {
    MESSAGES, FACTS, PENDING_FACTS, FACT_LINKS, DETECTED_CONTRADICTIONS,
    CORRECTIONS, EPISODES, EPISODE_MESSAGES, EPISODE_PATTERNS,
    OBSERVATIONS, OBSERVATION_REFLECTIONS, COMPANION_OBSERVATIONS,
    SYNTHESIZED_EVENTS, SYNTHESIZED_PARAGRAPHS, DAILY_SUMMARIES,
    RELATIONSHIPS, COMPANION_GOALS, COMPANION_GOAL_STEPS,
    COMPANION_AUTONOMOUS_TASKS, COMPANION_OPINIONS, COMPANION_JOURNAL,
    COMPANION_REMINDERS, CURIOSITY_THREADS, INTERACTION_OUTCOMES,
    INTERNAL_STATE_VALUES, RELATIONSHIP_EVAL, SCENE_STATE, STATE,
    CONVERSATION_PATTERNS, CONVERSATION_PATTERNS_STATS, EMOTION_ANALYTICS,
    USER_STATE, USER_PREFERENCES, FEED_POSTS, UPCOMING_EVENTS,
    BENCHMARK_QUESTIONS, BENCHMARK_RESULTS, BENCHMARK_RUNS,
    IMAGE_GENERATION_REQUESTS, DAILY_COST_SUMMARY, BUDGET_LIMITS,
    OPENROUTER_USAGE, FIREWORKS_USAGE, OPENAI_USAGE, GOOGLE_API_USAGE,
    HEDRA_USAGE, RUNCOMFY_USAGE, TWILIO_USAGE, ERROR_LOG,
}
