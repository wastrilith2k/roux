"""
Centralized DDL (CREATE TABLE / CREATE INDEX) for every table in the system.

Two entry points:
- get_public_schema_ddl()           -> DDL for shared tables in ``public``
- get_user_schema_ddl(schema_name)  -> DDL for all per-user tables, qualified
                                       with the given schema name

All table names come from ``tables.py`` constants so they stay in sync.
"""

from src.database import tables as T


# ---------------------------------------------------------------------------
# Public schema DDL
# ---------------------------------------------------------------------------

def get_public_schema_ddl() -> str:
    """Return DDL for all shared tables in the ``public`` schema."""
    return f"""
-- ============================================================================
-- PUBLIC SCHEMA TABLES
-- ============================================================================

-- Authentication: users
CREATE TABLE IF NOT EXISTS public.{T.USERS} (
    id SERIAL PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

-- Authentication: sessions
CREATE TABLE IF NOT EXISTS public.{T.SESSIONS} (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    session_token TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    FOREIGN KEY (user_id) REFERENCES public.{T.USERS} (id)
);

-- User profiles (shared reference for all schemas)
CREATE TABLE IF NOT EXISTS public.{T.USER_PROFILES} (
    email TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP
);

-- User-companion ownership
CREATE TABLE IF NOT EXISTS public.{T.USER_COMPANIONS} (
    id SERIAL PRIMARY KEY,
    user_email TEXT NOT NULL REFERENCES public.{T.USER_PROFILES}(email),
    companion_id TEXT NOT NULL,
    display_name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_email, companion_id)
);

-- Third-party service links (shared reference data)
CREATE TABLE IF NOT EXISTS public.{T.SERVICE_LINKS} (
    service_name TEXT PRIMARY KEY,
    billing_url TEXT NOT NULL,
    usage_url TEXT,
    api_dashboard_url TEXT,
    support_url TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


# ---------------------------------------------------------------------------
# User schema DDL
# ---------------------------------------------------------------------------

def get_user_schema_ddl(schema: str) -> str:
    """Return DDL for all per-user tables, qualified with *schema*.

    Every table name is prefixed with ``schema.`` so the caller can execute
    the result against a specific user schema.  Foreign-key references that
    would cross into the public schema (e.g. ``REFERENCES user_profiles``)
    are intentionally omitted to avoid cross-schema dependencies.
    """
    return "\n".join([
        _ddl_user_state(schema),
        _ddl_messages(schema),
        _ddl_facts(schema),
        _ddl_pending_facts(schema),
        _ddl_fact_links(schema),
        _ddl_detected_contradictions(schema),
        _ddl_corrections(schema),
        _ddl_episodes(schema),
        _ddl_episode_messages(schema),
        _ddl_episode_patterns(schema),
        _ddl_observations(schema),
        _ddl_observation_reflections(schema),
        _ddl_companion_observations(schema),
        _ddl_synthesized_events(schema),
        _ddl_synthesized_paragraphs(schema),
        _ddl_daily_summaries(schema),
        _ddl_relationships(schema),
        _ddl_companion_goals(schema),
        _ddl_companion_goal_steps(schema),
        _ddl_companion_autonomous_tasks(schema),
        _ddl_companion_opinions(schema),
        _ddl_companion_journal(schema),
        _ddl_companion_reminders(schema),
        _ddl_curiosity_threads(schema),
        _ddl_interaction_outcomes(schema),
        _ddl_internal_state_values(schema),
        _ddl_relationship_eval(schema),
        _ddl_scene_state(schema),
        _ddl_state(schema),
        _ddl_conversation_patterns(schema),
        _ddl_conversation_patterns_stats(schema),
        _ddl_emotion_analytics(schema),
        _ddl_user_preferences(schema),
        _ddl_feed_posts(schema),
        _ddl_upcoming_events(schema),
        _ddl_benchmark_questions(schema),
        _ddl_benchmark_runs(schema),
        _ddl_benchmark_results(schema),
        _ddl_image_generation_requests(schema),
        _ddl_daily_cost_summary(schema),
        _ddl_budget_limits(schema),
        _ddl_fireworks_usage(schema),
        _ddl_openai_usage(schema),
        _ddl_google_api_usage(schema),
        _ddl_hedra_usage(schema),
        _ddl_runcomfy_usage(schema),
        _ddl_twilio_usage(schema),
        _ddl_error_log(schema),
    ])


# ---------------------------------------------------------------------------
# Per-table DDL helpers  (private)
# ---------------------------------------------------------------------------

def _ddl_user_state(s: str) -> str:
    return f"""
-- User state (per companion)
CREATE TABLE IF NOT EXISTS {s}.{T.USER_STATE} (
    email TEXT NOT NULL,
    companion_id TEXT DEFAULT 'default',
    closeness_score INTEGER DEFAULT 15,
    romance_level REAL DEFAULT 0,
    romance_enabled BOOLEAN DEFAULT FALSE,
    romance_decision TEXT,
    emotion_profile TEXT DEFAULT 'Guarded',
    last_negative_event TIMESTAMP,
    cooldown_active BOOLEAN DEFAULT FALSE,
    badgering_count INTEGER DEFAULT 0,
    last_message_time TIMESTAMP,
    attraction_cue_count INTEGER DEFAULT 0,
    attraction_latent_state BOOLEAN DEFAULT FALSE,
    sms_preference TEXT DEFAULT 'good_morning',
    PRIMARY KEY (email, companion_id)
);
"""


def _ddl_messages(s: str) -> str:
    return f"""
-- Messages
CREATE TABLE IF NOT EXISTS {s}.{T.MESSAGES} (
    id SERIAL PRIMARY KEY,
    email TEXT NOT NULL,
    companion_id TEXT DEFAULT 'default',
    sender_name TEXT NOT NULL,
    message_text TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    sentiment_score FLOAT,
    closeness_after INTEGER,
    model_used TEXT,
    romance_level FLOAT,
    source TEXT DEFAULT 'chat',
    message_type TEXT DEFAULT 'normal',
    conversation_id INTEGER
);

CREATE INDEX IF NOT EXISTS idx_messages_email ON {s}.{T.MESSAGES}(email);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON {s}.{T.MESSAGES}(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON {s}.{T.MESSAGES}(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_companion_id ON {s}.{T.MESSAGES}(companion_id);
"""


def _ddl_facts(s: str) -> str:
    return f"""
-- Facts (subject-predicate-object knowledge)
CREATE TABLE IF NOT EXISTS {s}.{T.FACTS} (
    id SERIAL PRIMARY KEY,
    subject TEXT NOT NULL,
    predicate TEXT,
    object TEXT,
    confidence FLOAT DEFAULT 0.7,
    importance INTEGER DEFAULT 5,
    temporal TEXT DEFAULT 'current',
    context TEXT,
    source TEXT DEFAULT 'conversation',
    user_email TEXT,
    message_id INTEGER,
    embedding VECTOR(1536),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    mention_count INTEGER DEFAULT 1,
    last_mentioned TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    archived_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_facts_subject ON {s}.{T.FACTS}(subject);
CREATE INDEX IF NOT EXISTS idx_facts_user_email ON {s}.{T.FACTS}(user_email);
CREATE INDEX IF NOT EXISTS idx_facts_importance ON {s}.{T.FACTS}(importance DESC);
CREATE INDEX IF NOT EXISTS idx_facts_archived ON {s}.{T.FACTS}(archived_at) WHERE archived_at IS NULL;
"""


def _ddl_pending_facts(s: str) -> str:
    return f"""
-- Pending facts (approval queue)
CREATE TABLE IF NOT EXISTS {s}.{T.PENDING_FACTS} (
    id SERIAL PRIMARY KEY,
    subject VARCHAR(255) NOT NULL,
    predicate VARCHAR(255),
    fact_text TEXT NOT NULL,
    category VARCHAR(50),
    confidence FLOAT DEFAULT 0.7,
    importance INTEGER DEFAULT 5,
    sensitivity VARCHAR(50) NOT NULL,
    sensitivity_reason TEXT,
    llm_review TEXT,
    llm_recommendation VARCHAR(20),
    status VARCHAR(20) DEFAULT 'pending',
    user_email VARCHAR(255),
    message_id INTEGER,
    source_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reviewed_at TIMESTAMP,
    reviewed_by VARCHAR(255),
    edit_note TEXT
);

CREATE INDEX IF NOT EXISTS idx_pending_facts_status ON {s}.{T.PENDING_FACTS}(status, user_email);
"""


def _ddl_fact_links(s: str) -> str:
    return f"""
-- Fact links (associative memory network)
CREATE TABLE IF NOT EXISTS {s}.{T.FACT_LINKS} (
    id SERIAL PRIMARY KEY,
    source_fact_id INTEGER NOT NULL,
    target_fact_id INTEGER NOT NULL,
    link_type VARCHAR(50) NOT NULL,
    strength FLOAT NOT NULL DEFAULT 0.5,
    context TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_fact_id, target_fact_id, link_type)
);

CREATE INDEX IF NOT EXISTS idx_fact_links_source ON {s}.{T.FACT_LINKS}(source_fact_id);
CREATE INDEX IF NOT EXISTS idx_fact_links_target ON {s}.{T.FACT_LINKS}(target_fact_id);
CREATE INDEX IF NOT EXISTS idx_fact_links_type ON {s}.{T.FACT_LINKS}(link_type);
"""


def _ddl_detected_contradictions(s: str) -> str:
    return f"""
-- Detected contradictions between facts
CREATE TABLE IF NOT EXISTS {s}.{T.DETECTED_CONTRADICTIONS} (
    id SERIAL PRIMARY KEY,
    subject TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    detection_method VARCHAR(50),
    string_similarity FLOAT,
    semantic_similarity FLOAT,
    surfaced BOOLEAN DEFAULT FALSE,
    surfaced_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_contradictions_subject ON {s}.{T.DETECTED_CONTRADICTIONS}(subject);
CREATE INDEX IF NOT EXISTS idx_contradictions_surfaced ON {s}.{T.DETECTED_CONTRADICTIONS}(surfaced) WHERE surfaced = FALSE;
"""


def _ddl_corrections(s: str) -> str:
    return f"""
-- User-reported corrections
CREATE TABLE IF NOT EXISTS {s}.{T.CORRECTIONS} (
    id SERIAL PRIMARY KEY,
    email TEXT,
    subject TEXT,
    wrong_claim TEXT,
    correct_info TEXT,
    correction_type TEXT,
    importance INTEGER DEFAULT 5,
    confidence FLOAT DEFAULT 0.7,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_corrections_subject ON {s}.{T.CORRECTIONS}(subject);
CREATE INDEX IF NOT EXISTS idx_corrections_email ON {s}.{T.CORRECTIONS}(email);
"""


def _ddl_episodes(s: str) -> str:
    return f"""
-- Conversation episodes
CREATE TABLE IF NOT EXISTS {s}.{T.EPISODES} (
    id SERIAL PRIMARY KEY,
    episode_id UUID UNIQUE NOT NULL,
    user_email TEXT NOT NULL,
    started_at TIMESTAMP NOT NULL,
    ended_at TIMESTAMP,
    topic TEXT,
    trigger TEXT,
    emotional_state TEXT,
    resolution TEXT,
    user_satisfaction FLOAT,
    companion_approach TEXT,
    summary_embedding JSONB,
    pattern_id UUID,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_episodes_user_email ON {s}.{T.EPISODES}(user_email);
CREATE INDEX IF NOT EXISTS idx_episodes_started_at ON {s}.{T.EPISODES}(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_topic ON {s}.{T.EPISODES}(topic);
CREATE INDEX IF NOT EXISTS idx_episodes_pattern_id ON {s}.{T.EPISODES}(pattern_id);
"""


def _ddl_episode_messages(s: str) -> str:
    return f"""
-- Episode-message links
CREATE TABLE IF NOT EXISTS {s}.{T.EPISODE_MESSAGES} (
    id SERIAL PRIMARY KEY,
    episode_id UUID NOT NULL,
    message_id INTEGER NOT NULL,
    turn_number INTEGER NOT NULL,
    speaker TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_episode_messages_episode_id ON {s}.{T.EPISODE_MESSAGES}(episode_id);
CREATE INDEX IF NOT EXISTS idx_episode_messages_message_id ON {s}.{T.EPISODE_MESSAGES}(message_id);
"""


def _ddl_episode_patterns(s: str) -> str:
    return f"""
-- Episode patterns (consolidated learnings)
CREATE TABLE IF NOT EXISTS {s}.{T.EPISODE_PATTERNS} (
    id SERIAL PRIMARY KEY,
    pattern_id UUID UNIQUE NOT NULL,
    pattern_name TEXT NOT NULL,
    description TEXT,
    successful_approach TEXT,
    pitfalls_to_avoid TEXT,
    topic_category TEXT,
    emotional_context TEXT,
    episode_count INTEGER DEFAULT 0,
    avg_satisfaction FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_episode_patterns_topic ON {s}.{T.EPISODE_PATTERNS}(topic_category);
CREATE INDEX IF NOT EXISTS idx_episode_patterns_emotional ON {s}.{T.EPISODE_PATTERNS}(emotional_context);
"""


def _ddl_observations(s: str) -> str:
    return f"""
-- Observations (compressed dated conversation summaries)
CREATE TABLE IF NOT EXISTS {s}.{T.OBSERVATIONS} (
    id SERIAL PRIMARY KEY,
    user_email TEXT NOT NULL,
    observation_date DATE NOT NULL,
    time_range TEXT,
    content TEXT NOT NULL,
    message_count INTEGER DEFAULT 0,
    raw_token_count INTEGER DEFAULT 0,
    compressed_token_count INTEGER DEFAULT 0,
    compression_ratio FLOAT DEFAULT 0,
    topics TEXT,
    emotional_tone TEXT,
    first_message_id INTEGER,
    last_message_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_observations_user_date ON {s}.{T.OBSERVATIONS}(user_email, observation_date DESC);
CREATE INDEX IF NOT EXISTS idx_observations_date ON {s}.{T.OBSERVATIONS}(observation_date DESC);
"""


def _ddl_observation_reflections(s: str) -> str:
    return f"""
-- Observation reflections (weekly/monthly consolidations)
CREATE TABLE IF NOT EXISTS {s}.{T.OBSERVATION_REFLECTIONS} (
    id SERIAL PRIMARY KEY,
    user_email TEXT NOT NULL,
    period_type TEXT NOT NULL DEFAULT 'weekly',
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    content TEXT NOT NULL,
    themes TEXT,
    observation_ids TEXT,
    observation_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_reflections_user_period ON {s}.{T.OBSERVATION_REFLECTIONS}(user_email, period_end DESC);
"""


def _ddl_companion_observations(s: str) -> str:
    return f"""
-- Companion observations (per companion instance)
CREATE TABLE IF NOT EXISTS {s}.{T.COMPANION_OBSERVATIONS} (
    id SERIAL PRIMARY KEY,
    companion_id TEXT NOT NULL DEFAULT 'default',
    user_email TEXT NOT NULL,
    observation_type TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_companion_observations_cid ON {s}.{T.COMPANION_OBSERVATIONS}(companion_id);
CREATE INDEX IF NOT EXISTS idx_companion_observations_user ON {s}.{T.COMPANION_OBSERVATIONS}(user_email, companion_id);
"""


def _ddl_synthesized_events(s: str) -> str:
    return f"""
-- Synthesized events (narrative event summaries)
CREATE TABLE IF NOT EXISTS {s}.{T.SYNTHESIZED_EVENTS} (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    subject VARCHAR(255) NOT NULL,
    title VARCHAR(500) NOT NULL,
    timeline JSONB,
    narrative TEXT,
    outcome TEXT,
    impact TEXT,
    source_message_ids INTEGER[],
    base_importance FLOAT DEFAULT 5.0,
    embedding_vec VECTOR(1536),
    user_email VARCHAR(255),
    is_consolidated BOOLEAN DEFAULT FALSE,
    consolidated_from INTEGER[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(event_type, subject, title, user_email)
);

CREATE INDEX IF NOT EXISTS idx_synth_events_subject ON {s}.{T.SYNTHESIZED_EVENTS}(subject);
CREATE INDEX IF NOT EXISTS idx_synth_events_type ON {s}.{T.SYNTHESIZED_EVENTS}(event_type);
CREATE INDEX IF NOT EXISTS idx_synth_events_updated ON {s}.{T.SYNTHESIZED_EVENTS}(updated_at DESC);
"""


def _ddl_synthesized_paragraphs(s: str) -> str:
    return f"""
-- Synthesized biography paragraphs
CREATE TABLE IF NOT EXISTS {s}.{T.SYNTHESIZED_PARAGRAPHS} (
    id SERIAL PRIMARY KEY,
    theme VARCHAR(100) NOT NULL,
    subject VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    fact_ids INTEGER[] NOT NULL,
    base_importance FLOAT NOT NULL,
    embedding_vec VECTOR(1536),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_email VARCHAR(255),
    UNIQUE(theme, subject, user_email)
);

CREATE INDEX IF NOT EXISTS idx_synth_subject ON {s}.{T.SYNTHESIZED_PARAGRAPHS}(subject);
CREATE INDEX IF NOT EXISTS idx_synth_theme ON {s}.{T.SYNTHESIZED_PARAGRAPHS}(theme);
"""


def _ddl_daily_summaries(s: str) -> str:
    return f"""
-- Daily summaries
CREATE TABLE IF NOT EXISTS {s}.{T.DAILY_SUMMARIES} (
    id SERIAL PRIMARY KEY,
    user_email VARCHAR(255) NOT NULL,
    summary_date DATE NOT NULL,
    content TEXT NOT NULL,
    message_count INTEGER,
    facts_learned INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_email, summary_date)
);

CREATE INDEX IF NOT EXISTS idx_daily_summaries_date ON {s}.{T.DAILY_SUMMARIES}(summary_date DESC);
"""


def _ddl_relationships(s: str) -> str:
    return f"""
-- Relationships between entities
CREATE TABLE IF NOT EXISTS {s}.{T.RELATIONSHIPS} (
    id SERIAL PRIMARY KEY,
    source_entity VARCHAR(255) NOT NULL,
    relationship_type VARCHAR(50) NOT NULL,
    target_entity VARCHAR(255) NOT NULL,
    confidence FLOAT DEFAULT 0.7,
    mention_count INTEGER DEFAULT 1,
    last_verified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    valid_from TIMESTAMP,
    valid_until TIMESTAMP,
    context TEXT,
    source_message_id INTEGER,
    user_email VARCHAR(255),
    is_primary BOOLEAN DEFAULT TRUE,
    contradiction_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_entity, relationship_type, target_entity, user_email)
);

CREATE INDEX IF NOT EXISTS idx_rel_source ON {s}.{T.RELATIONSHIPS}(source_entity);
CREATE INDEX IF NOT EXISTS idx_rel_target ON {s}.{T.RELATIONSHIPS}(target_entity);
CREATE INDEX IF NOT EXISTS idx_rel_type ON {s}.{T.RELATIONSHIPS}(relationship_type);
CREATE INDEX IF NOT EXISTS idx_rel_valid ON {s}.{T.RELATIONSHIPS}(valid_until) WHERE valid_until IS NULL;
"""


def _ddl_companion_goals(s: str) -> str:
    return f"""
-- Companion goals
CREATE TABLE IF NOT EXISTS {s}.{T.COMPANION_GOALS} (
    id VARCHAR(50) PRIMARY KEY,
    user_email VARCHAR(255),
    goal TEXT NOT NULL,
    motivation TEXT,
    category VARCHAR(50),
    progress REAL DEFAULT 0.0,
    status VARCHAR(20) DEFAULT 'active',
    actions_taken JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deadline TIMESTAMP,
    priority REAL DEFAULT 0.5,
    goal_mode VARCHAR(20) DEFAULT 'doing',
    energy_cost VARCHAR(20) DEFAULT 'medium'
);

CREATE INDEX IF NOT EXISTS idx_goals_status ON {s}.{T.COMPANION_GOALS}(user_email, status);
CREATE INDEX IF NOT EXISTS idx_goals_category ON {s}.{T.COMPANION_GOALS}(user_email, category);
"""


def _ddl_companion_goal_steps(s: str) -> str:
    return f"""
-- Companion goal steps (actionable steps toward goals)
CREATE TABLE IF NOT EXISTS {s}.{T.COMPANION_GOAL_STEPS} (
    id VARCHAR(100) PRIMARY KEY,
    goal_id VARCHAR(50) NOT NULL,
    user_email VARCHAR(255),
    step_number INTEGER NOT NULL,
    description TEXT NOT NULL,
    action_type VARCHAR(50),
    parameters JSONB DEFAULT '{{}}'::jsonb,
    status VARCHAR(20) DEFAULT 'pending',
    depends_on JSONB DEFAULT '[]'::jsonb,
    wait_for TEXT,
    outcome TEXT,
    outcome_quality VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3
);

CREATE INDEX IF NOT EXISTS idx_goal_steps_goal ON {s}.{T.COMPANION_GOAL_STEPS}(goal_id);
CREATE INDEX IF NOT EXISTS idx_goal_steps_status ON {s}.{T.COMPANION_GOAL_STEPS}(status);
"""


def _ddl_companion_autonomous_tasks(s: str) -> str:
    return f"""
-- Companion autonomous tasks
CREATE TABLE IF NOT EXISTS {s}.{T.COMPANION_AUTONOMOUS_TASKS} (
    task_id TEXT PRIMARY KEY,
    user_id TEXT,
    task_name TEXT,
    status TEXT DEFAULT 'pending',
    narrative TEXT,
    mood_delta FLOAT DEFAULT 0,
    energy_delta FLOAT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_auto_tasks_status ON {s}.{T.COMPANION_AUTONOMOUS_TASKS}(status);
"""


def _ddl_companion_opinions(s: str) -> str:
    return f"""
-- Companion opinions
CREATE TABLE IF NOT EXISTS {s}.{T.COMPANION_OPINIONS} (
    id SERIAL PRIMARY KEY,
    user_email VARCHAR(255),
    topic VARCHAR(255),
    opinion TEXT,
    confidence REAL,
    evidence_count INTEGER DEFAULT 1,
    category VARCHAR(50),
    evidence_summary TEXT,
    formed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_email, topic)
);

CREATE INDEX IF NOT EXISTS idx_opinions_category ON {s}.{T.COMPANION_OPINIONS}(user_email, category);
CREATE INDEX IF NOT EXISTS idx_opinions_confidence ON {s}.{T.COMPANION_OPINIONS}(user_email, confidence DESC);
"""


def _ddl_companion_journal(s: str) -> str:
    return f"""
-- Companion journal
CREATE TABLE IF NOT EXISTS {s}.{T.COMPANION_JOURNAL} (
    id SERIAL PRIMARY KEY,
    user_email VARCHAR(255),
    entry_date DATE,
    entry_type VARCHAR(50),
    content TEXT,
    insights JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_journal_date ON {s}.{T.COMPANION_JOURNAL}(entry_date DESC);
CREATE INDEX IF NOT EXISTS idx_journal_type ON {s}.{T.COMPANION_JOURNAL}(entry_type, user_email);
"""


def _ddl_companion_reminders(s: str) -> str:
    return f"""
-- Companion reminders
CREATE TABLE IF NOT EXISTS {s}.{T.COMPANION_REMINDERS} (
    id SERIAL PRIMARY KEY,
    companion_id TEXT NOT NULL DEFAULT 'default',
    user_email TEXT NOT NULL,
    reminder_text TEXT NOT NULL,
    remind_at TIMESTAMP NOT NULL,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_companion_reminders_status ON {s}.{T.COMPANION_REMINDERS}(companion_id, status);
CREATE INDEX IF NOT EXISTS idx_companion_reminders_remind_at ON {s}.{T.COMPANION_REMINDERS}(remind_at);
"""


def _ddl_curiosity_threads(s: str) -> str:
    return f"""
-- Curiosity threads (topics the companion is curious about)
CREATE TABLE IF NOT EXISTS {s}.{T.CURIOSITY_THREADS} (
    id SERIAL PRIMARY KEY,
    topic TEXT NOT NULL,
    category VARCHAR(50),
    trigger_message TEXT,
    questions JSONB DEFAULT '[]'::jsonb,
    urgency FLOAT DEFAULT 0.3,
    source VARCHAR(50) DEFAULT 'conversation',
    times_asked INTEGER DEFAULT 0,
    last_discussed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_curiosity_urgency ON {s}.{T.CURIOSITY_THREADS}(urgency DESC);
"""


def _ddl_interaction_outcomes(s: str) -> str:
    return f"""
-- Interaction outcomes
CREATE TABLE IF NOT EXISTS {s}.{T.INTERACTION_OUTCOMES} (
    id SERIAL PRIMARY KEY,
    user_email VARCHAR(255) NOT NULL,
    companion_message_id INTEGER,
    user_response_id INTEGER,
    companion_message_text TEXT,
    user_response_text TEXT,
    companion_action_type VARCHAR(50),
    companion_topic VARCHAR(255),
    engagement_level VARCHAR(20),
    topic_continued BOOLEAN DEFAULT FALSE,
    emotional_resonance FLOAT,
    analysis_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_outcomes_email ON {s}.{T.INTERACTION_OUTCOMES}(user_email);
CREATE INDEX IF NOT EXISTS idx_outcomes_action ON {s}.{T.INTERACTION_OUTCOMES}(companion_action_type);
CREATE INDEX IF NOT EXISTS idx_outcomes_date ON {s}.{T.INTERACTION_OUTCOMES}(created_at DESC);
"""


def _ddl_internal_state_values(s: str) -> str:
    return f"""
-- Hidden internal state values (value inference)
CREATE TABLE IF NOT EXISTS {s}.{T.INTERNAL_STATE_VALUES} (
    id SERIAL PRIMARY KEY,
    category VARCHAR(50) NOT NULL,
    key_hash VARCHAR(16) NOT NULL,
    value_data JSONB NOT NULL,
    confidence FLOAT DEFAULT 0.7,
    evidence_count INTEGER DEFAULT 1,
    last_updated TIMESTAMP DEFAULT NOW(),
    UNIQUE(category, key_hash)
);
"""


def _ddl_relationship_eval(s: str) -> str:
    return f"""
-- Hidden relationship evaluation
CREATE TABLE IF NOT EXISTS {s}.{T.RELATIONSHIP_EVAL} (
    id SERIAL PRIMARY KEY,
    user_email VARCHAR(255) UNIQUE NOT NULL,
    evaluation JSONB NOT NULL,
    previous_evaluation JSONB,
    evaluated_at TIMESTAMP DEFAULT NOW(),
    evaluation_count INTEGER DEFAULT 1
);
"""


def _ddl_scene_state(s: str) -> str:
    return f"""
-- Scene state (roleplay/virtual scene persistence)
CREATE TABLE IF NOT EXISTS {s}.{T.SCENE_STATE} (
    id SERIAL PRIMARY KEY,
    user_email TEXT NOT NULL,
    companion_id TEXT DEFAULT 'default',
    scene_data JSONB,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_email, companion_id)
);
"""


def _ddl_state(s: str) -> str:
    return f"""
-- Generic key-value state store
CREATE TABLE IF NOT EXISTS {s}.{T.STATE} (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_state_key ON {s}.{T.STATE}(key);
"""


def _ddl_conversation_patterns(s: str) -> str:
    return f"""
-- Conversation patterns
CREATE TABLE IF NOT EXISTS {s}.{T.CONVERSATION_PATTERNS} (
    id SERIAL PRIMARY KEY,
    email TEXT NOT NULL,
    user_length INTEGER,
    response_length INTEGER,
    closeness INTEGER,
    pattern_type TEXT DEFAULT 'exchange',
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_patterns_email ON {s}.{T.CONVERSATION_PATTERNS}(email);
CREATE INDEX IF NOT EXISTS idx_patterns_recorded_at ON {s}.{T.CONVERSATION_PATTERNS}(recorded_at DESC);
"""


def _ddl_conversation_patterns_stats(s: str) -> str:
    return f"""
-- Conversation patterns statistics (aggregated)
CREATE TABLE IF NOT EXISTS {s}.{T.CONVERSATION_PATTERNS_STATS} (
    email TEXT PRIMARY KEY,
    total_exchanges INTEGER DEFAULT 0,
    avg_user_length FLOAT DEFAULT 0,
    avg_response_length FLOAT DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def _ddl_emotion_analytics(s: str) -> str:
    return f"""
-- Emotion analytics
CREATE TABLE IF NOT EXISTS {s}.{T.EMOTION_ANALYTICS} (
    id SERIAL PRIMARY KEY,
    email TEXT NOT NULL,
    emotion TEXT,
    confidence FLOAT,
    intensity FLOAT,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_emotion_email ON {s}.{T.EMOTION_ANALYTICS}(email);
CREATE INDEX IF NOT EXISTS idx_emotion_recorded_at ON {s}.{T.EMOTION_ANALYTICS}(recorded_at DESC);
"""


def _ddl_user_preferences(s: str) -> str:
    return f"""
-- User preferences (learned)
CREATE TABLE IF NOT EXISTS {s}.{T.USER_PREFERENCES} (
    id SERIAL PRIMARY KEY,
    email TEXT NOT NULL,
    preference_type TEXT,
    preference_value TEXT,
    learned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_prefs_email ON {s}.{T.USER_PREFERENCES}(email);
CREATE INDEX IF NOT EXISTS idx_prefs_type ON {s}.{T.USER_PREFERENCES}(preference_type);
"""


def _ddl_feed_posts(s: str) -> str:
    return f"""
-- Microblog / feed posts
CREATE TABLE IF NOT EXISTS {s}.{T.FEED_POSTS} (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    mood TEXT,
    tags TEXT,
    is_public BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_feed_posts_timestamp ON {s}.{T.FEED_POSTS}(timestamp DESC);
"""


def _ddl_upcoming_events(s: str) -> str:
    return f"""
-- Upcoming events
CREATE TABLE IF NOT EXISTS {s}.{T.UPCOMING_EVENTS} (
    id SERIAL PRIMARY KEY,
    user_email TEXT NOT NULL,
    event_type TEXT NOT NULL,
    description TEXT NOT NULL,
    scheduled_time TIMESTAMP,
    created_at TIMESTAMP NOT NULL,
    status TEXT DEFAULT 'planned',
    participants TEXT,
    location TEXT,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_user_status ON {s}.{T.UPCOMING_EVENTS}(user_email, status);
CREATE INDEX IF NOT EXISTS idx_events_scheduled_time ON {s}.{T.UPCOMING_EVENTS}(scheduled_time);
"""


def _ddl_benchmark_questions(s: str) -> str:
    return f"""
-- Benchmark questions
CREATE TABLE IF NOT EXISTS {s}.{T.BENCHMARK_QUESTIONS} (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    expected_answer TEXT NOT NULL,
    expected_keywords TEXT,
    negative_keywords TEXT,
    category TEXT NOT NULL,
    difficulty TEXT DEFAULT 'medium',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_benchmark_questions_category ON {s}.{T.BENCHMARK_QUESTIONS}(category);
"""


def _ddl_benchmark_runs(s: str) -> str:
    return f"""
-- Benchmark runs
CREATE TABLE IF NOT EXISTS {s}.{T.BENCHMARK_RUNS} (
    id SERIAL PRIMARY KEY,
    run_type TEXT NOT NULL,
    config_snapshot JSONB,
    overall_score FLOAT,
    category_scores JSONB,
    total_cost FLOAT DEFAULT 0,
    total_time_seconds FLOAT DEFAULT 0,
    question_count INTEGER DEFAULT 0,
    observations_enabled BOOLEAN DEFAULT FALSE,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def _ddl_benchmark_results(s: str) -> str:
    return f"""
-- Benchmark results (per question per run)
CREATE TABLE IF NOT EXISTS {s}.{T.BENCHMARK_RESULTS} (
    id SERIAL PRIMARY KEY,
    run_id INTEGER NOT NULL,
    question_id INTEGER NOT NULL,
    context_retrieved TEXT,
    context_sources JSONB,
    response TEXT,
    accuracy_score FLOAT,
    confabulation_score FLOAT,
    completeness_score FLOAT,
    context_utilization_score FLOAT,
    keyword_hits INTEGER DEFAULT 0,
    keyword_misses INTEGER DEFAULT 0,
    negative_keyword_hits INTEGER DEFAULT 0,
    context_build_time_ms INTEGER,
    response_time_ms INTEGER,
    judge_time_ms INTEGER,
    cost_estimate FLOAT DEFAULT 0,
    judge_reasoning TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_benchmark_results_run ON {s}.{T.BENCHMARK_RESULTS}(run_id);
CREATE INDEX IF NOT EXISTS idx_benchmark_results_question ON {s}.{T.BENCHMARK_RESULTS}(question_id);
"""


def _ddl_image_generation_requests(s: str) -> str:
    return f"""
-- Image generation requests
CREATE TABLE IF NOT EXISTS {s}.{T.IMAGE_GENERATION_REQUESTS} (
    id SERIAL PRIMARY KEY,
    task_id TEXT UNIQUE NOT NULL,
    email TEXT,
    prompt TEXT,
    workflow_type TEXT,
    width INTEGER DEFAULT 1024,
    height INTEGER DEFAULT 1024,
    status TEXT DEFAULT 'pending',
    runcomfy_request_id TEXT,
    runcomfy_url TEXT,
    cloudinary_url TEXT,
    downloaded_at TIMESTAMP,
    uploaded_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_image_gen_status ON {s}.{T.IMAGE_GENERATION_REQUESTS}(status);
CREATE INDEX IF NOT EXISTS idx_image_gen_task ON {s}.{T.IMAGE_GENERATION_REQUESTS}(task_id);
"""


def _ddl_daily_cost_summary(s: str) -> str:
    return f"""
-- Daily cost summary aggregation
CREATE TABLE IF NOT EXISTS {s}.{T.DAILY_COST_SUMMARY} (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    user_id TEXT NOT NULL,
    fireworks_calls INTEGER DEFAULT 0,
    fireworks_tokens_in INTEGER DEFAULT 0,
    fireworks_tokens_out INTEGER DEFAULT 0,
    fireworks_cost_usd REAL DEFAULT 0,
    openai_calls INTEGER DEFAULT 0,
    openai_tokens_in INTEGER DEFAULT 0,
    openai_tokens_out INTEGER DEFAULT 0,
    openai_voice_minutes REAL DEFAULT 0,
    openai_cost_usd REAL DEFAULT 0,
    hedra_videos INTEGER DEFAULT 0,
    hedra_minutes REAL DEFAULT 0,
    hedra_cost_usd REAL DEFAULT 0,
    runcomfy_images INTEGER DEFAULT 0,
    runcomfy_cost_usd REAL DEFAULT 0,
    twilio_sms_sent INTEGER DEFAULT 0,
    twilio_sms_received INTEGER DEFAULT 0,
    twilio_cost_usd REAL DEFAULT 0,
    google_api_calls INTEGER DEFAULT 0,
    google_quota_used INTEGER DEFAULT 0,
    google_cost_usd REAL DEFAULT 0,
    total_cost_usd REAL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, user_id)
);

CREATE INDEX IF NOT EXISTS idx_daily_summary_user_date ON {s}.{T.DAILY_COST_SUMMARY}(user_id, date);
"""


def _ddl_budget_limits(s: str) -> str:
    return f"""
-- Budget limits per user
CREATE TABLE IF NOT EXISTS {s}.{T.BUDGET_LIMITS} (
    user_id TEXT PRIMARY KEY,
    fireworks_monthly_limit_usd REAL DEFAULT 150.00,
    openai_monthly_limit_usd REAL DEFAULT 50.00,
    hedra_monthly_limit_usd REAL DEFAULT 100.00,
    runcomfy_monthly_limit_usd REAL DEFAULT 25.00,
    twilio_monthly_limit_usd REAL DEFAULT 15.00,
    google_monthly_limit_usd REAL DEFAULT 0.00,
    total_monthly_limit_usd REAL DEFAULT 750.00,
    warning_threshold_pct REAL DEFAULT 0.75,
    critical_threshold_pct REAL DEFAULT 0.90,
    enable_hard_stop BOOLEAN DEFAULT FALSE,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def _ddl_fireworks_usage(s: str) -> str:
    return f"""
-- Fireworks.ai usage tracking
CREATE TABLE IF NOT EXISTS {s}.{T.FIREWORKS_USAGE} (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id TEXT NOT NULL,
    model TEXT DEFAULT 'llama-v3p1-70b-instruct',
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    endpoint TEXT,
    response_time_ms INTEGER,
    error BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_fireworks_user_timestamp ON {s}.{T.FIREWORKS_USAGE}(user_id, timestamp);
"""


def _ddl_openai_usage(s: str) -> str:
    return f"""
-- OpenAI usage tracking
CREATE TABLE IF NOT EXISTS {s}.{T.OPENAI_USAGE} (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id TEXT NOT NULL,
    model TEXT DEFAULT 'gpt-4o-mini',
    service_type TEXT NOT NULL,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    audio_seconds REAL DEFAULT 0,
    characters INTEGER DEFAULT 0,
    cost_usd REAL NOT NULL,
    error BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_openai_user_timestamp ON {s}.{T.OPENAI_USAGE}(user_id, timestamp);
"""


def _ddl_google_api_usage(s: str) -> str:
    return f"""
-- Google API usage tracking
CREATE TABLE IF NOT EXISTS {s}.{T.GOOGLE_API_USAGE} (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id TEXT NOT NULL,
    service TEXT NOT NULL,
    endpoint TEXT,
    method TEXT,
    quota_cost INTEGER DEFAULT 1,
    response_time_ms INTEGER,
    error BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_google_user_timestamp ON {s}.{T.GOOGLE_API_USAGE}(user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_google_service ON {s}.{T.GOOGLE_API_USAGE}(service);
"""


def _ddl_hedra_usage(s: str) -> str:
    return f"""
-- Hedra avatar video usage tracking
CREATE TABLE IF NOT EXISTS {s}.{T.HEDRA_USAGE} (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id TEXT NOT NULL,
    job_id TEXT UNIQUE NOT NULL,
    message_id TEXT,
    audio_input_path TEXT,
    portrait_image_path TEXT,
    video_output_path TEXT,
    duration_seconds INTEGER,
    cost_usd REAL NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT,
    completed_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_hedra_user_timestamp ON {s}.{T.HEDRA_USAGE}(user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_hedra_status ON {s}.{T.HEDRA_USAGE}(status);
"""


def _ddl_runcomfy_usage(s: str) -> str:
    return f"""
-- RunComfy image generation usage tracking
CREATE TABLE IF NOT EXISTS {s}.{T.RUNCOMFY_USAGE} (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id TEXT NOT NULL,
    workflow_id TEXT NOT NULL,
    workflow_type TEXT NOT NULL,
    prompt TEXT,
    image_url TEXT,
    generation_time_seconds INTEGER,
    cost_usd REAL NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_runcomfy_user_timestamp ON {s}.{T.RUNCOMFY_USAGE}(user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_runcomfy_status ON {s}.{T.RUNCOMFY_USAGE}(status);
"""


def _ddl_twilio_usage(s: str) -> str:
    return f"""
-- Twilio SMS usage tracking
CREATE TABLE IF NOT EXISTS {s}.{T.TWILIO_USAGE} (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id TEXT NOT NULL,
    message_sid TEXT UNIQUE NOT NULL,
    direction TEXT NOT NULL,
    to_number TEXT,
    from_number TEXT,
    message_body TEXT,
    status TEXT,
    cost_usd REAL NOT NULL,
    error_code TEXT
);

CREATE INDEX IF NOT EXISTS idx_twilio_user_timestamp ON {s}.{T.TWILIO_USAGE}(user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_twilio_direction ON {s}.{T.TWILIO_USAGE}(direction);
"""


def _ddl_error_log(s: str) -> str:
    return f"""
-- Error log
CREATE TABLE IF NOT EXISTS {s}.{T.ERROR_LOG} (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    module VARCHAR(100),
    function_name VARCHAR(100),
    companion_id VARCHAR(50),
    error_type VARCHAR(200),
    error_message TEXT,
    stack_trace TEXT,
    context TEXT,
    resolved BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_error_log_timestamp ON {s}.{T.ERROR_LOG}(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_error_log_companion ON {s}.{T.ERROR_LOG}(companion_id, timestamp DESC);
"""
