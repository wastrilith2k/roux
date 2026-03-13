-- ============================================================================
-- ESME AI - UNIFIED COST TRACKING DATABASE SCHEMA
-- ============================================================================
-- Purpose: Track costs across all external services (Fireworks, OpenAI,
--          RunComfy, Twilio, Hedra, Google APIs)
-- Created: October 19, 2025
-- ============================================================================

-- Fireworks.ai usage tracking
CREATE TABLE IF NOT EXISTS fireworks_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    user_id TEXT NOT NULL,
    model TEXT DEFAULT 'llama-v3p1-70b-instruct',
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    endpoint TEXT, -- 'chat', 'completion', etc.
    response_time_ms INTEGER,
    error BOOLEAN DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_fireworks_user_timestamp ON fireworks_usage(user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_fireworks_date ON fireworks_usage(DATE(timestamp));

-- OpenAI usage tracking
CREATE TABLE IF NOT EXISTS openai_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    user_id TEXT NOT NULL,
    model TEXT DEFAULT 'gpt-4o-mini',
    service_type TEXT NOT NULL, -- 'tool_detection', 'chat', 'tts', 'realtime_voice'
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    audio_seconds REAL DEFAULT 0, -- For Realtime Voice API
    characters INTEGER DEFAULT 0, -- For TTS
    cost_usd REAL NOT NULL,
    error BOOLEAN DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_openai_user_timestamp ON openai_usage(user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_openai_date ON openai_usage(DATE(timestamp));

-- Hedra avatar video usage tracking
CREATE TABLE IF NOT EXISTS hedra_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    user_id TEXT NOT NULL,
    job_id TEXT UNIQUE NOT NULL,
    message_id TEXT, -- Link to conversation message
    audio_input_path TEXT,
    portrait_image_path TEXT,
    video_output_path TEXT,
    duration_seconds INTEGER,
    cost_usd REAL NOT NULL,
    status TEXT NOT NULL, -- 'pending', 'processing', 'completed', 'failed'
    error_message TEXT,
    completed_at DATETIME
);

CREATE INDEX IF NOT EXISTS idx_hedra_user_timestamp ON hedra_usage(user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_hedra_date ON hedra_usage(DATE(timestamp));
CREATE INDEX IF NOT EXISTS idx_hedra_status ON hedra_usage(status);

-- RunComfy image generation usage tracking
CREATE TABLE IF NOT EXISTS runcomfy_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    user_id TEXT NOT NULL,
    workflow_id TEXT NOT NULL,
    workflow_type TEXT NOT NULL, -- 'personal', 'general'
    prompt TEXT,
    image_url TEXT,
    generation_time_seconds INTEGER,
    cost_usd REAL NOT NULL,
    status TEXT NOT NULL, -- 'success', 'failed', 'timeout'
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_runcomfy_user_timestamp ON runcomfy_usage(user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_runcomfy_date ON runcomfy_usage(DATE(timestamp));
CREATE INDEX IF NOT EXISTS idx_runcomfy_status ON runcomfy_usage(status);

-- Twilio SMS usage tracking
CREATE TABLE IF NOT EXISTS twilio_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    user_id TEXT NOT NULL,
    message_sid TEXT UNIQUE NOT NULL,
    direction TEXT NOT NULL, -- 'outbound', 'inbound'
    to_number TEXT,
    from_number TEXT,
    message_body TEXT,
    status TEXT,
    cost_usd REAL NOT NULL, -- $0.10 for outbound, $0.00 for inbound
    error_code TEXT
);

CREATE INDEX IF NOT EXISTS idx_twilio_user_timestamp ON twilio_usage(user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_twilio_date ON twilio_usage(DATE(timestamp));
CREATE INDEX IF NOT EXISTS idx_twilio_direction ON twilio_usage(direction);

-- Google Workspace API usage tracking
CREATE TABLE IF NOT EXISTS google_api_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    user_id TEXT NOT NULL,
    service TEXT NOT NULL, -- 'gmail', 'calendar', 'drive', 'chat', 'tasks'
    endpoint TEXT,
    method TEXT, -- 'GET', 'POST', 'PUT', 'DELETE'
    quota_cost INTEGER DEFAULT 1, -- Most calls cost 1 quota unit
    response_time_ms INTEGER,
    error BOOLEAN DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_google_user_timestamp ON google_api_usage(user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_google_date ON google_api_usage(DATE(timestamp));
CREATE INDEX IF NOT EXISTS idx_google_service ON google_api_usage(service);

-- Daily cost summary aggregation
CREATE TABLE IF NOT EXISTS daily_cost_summary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    user_id TEXT NOT NULL,

    -- Fireworks.ai
    fireworks_calls INTEGER DEFAULT 0,
    fireworks_tokens_in INTEGER DEFAULT 0,
    fireworks_tokens_out INTEGER DEFAULT 0,
    fireworks_cost_usd REAL DEFAULT 0,

    -- OpenAI
    openai_calls INTEGER DEFAULT 0,
    openai_tokens_in INTEGER DEFAULT 0,
    openai_tokens_out INTEGER DEFAULT 0,
    openai_voice_minutes REAL DEFAULT 0,
    openai_cost_usd REAL DEFAULT 0,

    -- Hedra
    hedra_videos INTEGER DEFAULT 0,
    hedra_minutes REAL DEFAULT 0,
    hedra_cost_usd REAL DEFAULT 0,

    -- RunComfy
    runcomfy_images INTEGER DEFAULT 0,
    runcomfy_cost_usd REAL DEFAULT 0,

    -- Twilio
    twilio_sms_sent INTEGER DEFAULT 0,
    twilio_sms_received INTEGER DEFAULT 0,
    twilio_cost_usd REAL DEFAULT 0,

    -- Google APIs
    google_api_calls INTEGER DEFAULT 0,
    google_quota_used INTEGER DEFAULT 0,
    google_cost_usd REAL DEFAULT 0,

    -- Totals
    total_cost_usd REAL DEFAULT 0,

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(date, user_id)
);

CREATE INDEX IF NOT EXISTS idx_daily_summary_user_date ON daily_cost_summary(user_id, date);

-- Budget limits per user
CREATE TABLE IF NOT EXISTS budget_limits (
    user_id TEXT PRIMARY KEY,

    -- Per-service monthly budgets
    fireworks_monthly_limit_usd REAL DEFAULT 150.00,
    openai_monthly_limit_usd REAL DEFAULT 50.00,
    hedra_monthly_limit_usd REAL DEFAULT 100.00,
    runcomfy_monthly_limit_usd REAL DEFAULT 25.00,
    twilio_monthly_limit_usd REAL DEFAULT 15.00,
    google_monthly_limit_usd REAL DEFAULT 0.00, -- Currently free

    -- Overall limits
    total_monthly_limit_usd REAL DEFAULT 750.00,

    -- Warning thresholds (percentage)
    warning_threshold_pct REAL DEFAULT 0.75, -- Alert at 75%
    critical_threshold_pct REAL DEFAULT 0.90, -- Critical at 90%

    -- Enable hard stop at limit
    enable_hard_stop BOOLEAN DEFAULT 0,

    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Alert/warning log
CREATE TABLE IF NOT EXISTS usage_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    alert_type TEXT NOT NULL, -- 'warning', 'critical', 'limit_reached'
    service TEXT NOT NULL, -- 'fireworks', 'openai', 'hedra', 'runcomfy', 'twilio', 'total'
    threshold_type TEXT NOT NULL, -- 'daily', 'monthly'
    current_amount_usd REAL NOT NULL,
    limit_amount_usd REAL NOT NULL,
    percentage REAL NOT NULL,
    message TEXT,
    acknowledged BOOLEAN DEFAULT 0,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_alerts_user_timestamp ON usage_alerts(user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged ON usage_alerts(acknowledged);

-- Third-party service links (for quick access to billing pages)
CREATE TABLE IF NOT EXISTS service_links (
    service_name TEXT PRIMARY KEY,
    billing_url TEXT NOT NULL,
    usage_url TEXT,
    api_dashboard_url TEXT,
    support_url TEXT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Insert default service links
INSERT OR REPLACE INTO service_links (service_name, billing_url, usage_url, api_dashboard_url, support_url) VALUES
('fireworks', 'https://fireworks.ai/account/billing', 'https://fireworks.ai/account/usage', 'https://fireworks.ai/account/api-keys', 'https://docs.fireworks.ai/'),
('openai', 'https://platform.openai.com/account/billing/overview', 'https://platform.openai.com/usage', 'https://platform.openai.com/api-keys', 'https://help.openai.com/'),
('hedra', 'https://www.hedra.com/account/billing', 'https://www.hedra.com/account/usage', 'https://www.hedra.com/api', 'https://docs.hedra.com/'),
('runcomfy', 'https://www.runcomfy.com/billing', 'https://www.runcomfy.com/usage', 'https://www.runcomfy.com/api', 'https://docs.runcomfy.com/'),
('twilio', 'https://console.twilio.com/us1/billing/manage-billing/billing-overview', 'https://console.twilio.com/us1/monitor/logs/sms', 'https://console.twilio.com/us1/develop/sms/overview', 'https://www.twilio.com/docs'),
('google', 'https://console.cloud.google.com/billing', 'https://console.cloud.google.com/apis/dashboard', 'https://console.cloud.google.com/apis/credentials', 'https://cloud.google.com/docs');

-- ============================================================================
-- HELPER VIEWS
-- ============================================================================

-- Current month summary per service
CREATE VIEW IF NOT EXISTS v_current_month_summary AS
SELECT
    user_id,
    SUM(fireworks_cost_usd) as fireworks_total,
    SUM(openai_cost_usd) as openai_total,
    SUM(hedra_cost_usd) as hedra_total,
    SUM(runcomfy_cost_usd) as runcomfy_total,
    SUM(twilio_cost_usd) as twilio_total,
    SUM(google_cost_usd) as google_total,
    SUM(total_cost_usd) as grand_total,
    COUNT(*) as days_tracked
FROM daily_cost_summary
WHERE strftime('%Y-%m', date) = strftime('%Y-%m', 'now')
GROUP BY user_id;

-- Today's summary
CREATE VIEW IF NOT EXISTS v_today_summary AS
SELECT
    user_id,
    fireworks_cost_usd,
    openai_cost_usd,
    hedra_cost_usd,
    runcomfy_cost_usd,
    twilio_cost_usd,
    google_cost_usd,
    total_cost_usd
FROM daily_cost_summary
WHERE date = DATE('now');

-- ============================================================================
-- INITIALIZATION
-- ============================================================================

-- Create default budget for admin user
INSERT OR IGNORE INTO budget_limits (user_id) VALUES ('admin');
