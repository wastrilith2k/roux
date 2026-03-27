"""
Centralized Model Configuration

Single source of truth for LLM model identifiers. All modules should import
from here instead of hardcoding model strings. When a provider deprecates a
model, only this file needs to change.
"""

import os

# ---------------------------------------------------------------------------
# Fireworks models
# ---------------------------------------------------------------------------

# Primary conversation model (env-overridable for zero-downtime swaps)
FIREWORKS_DEFAULT_MODEL = os.getenv(
    "FIREWORKS_MODEL",
    "accounts/fireworks/models/kimi-k2p5-instruct",
)

# Fallback conversation model (used when primary fails)
FIREWORKS_FALLBACK_MODEL = os.getenv(
    "FIREWORKS_FALLBACK_MODEL",
    FIREWORKS_DEFAULT_MODEL,
)

# Background/autonomy tasks model (cheaper or same as default)
FIREWORKS_BACKGROUND_MODEL = os.getenv(
    "AUTONOMY_MODEL",
    FIREWORKS_DEFAULT_MODEL,
)

# Short name for display purposes (e.g. status commands)
FIREWORKS_DEFAULT_MODEL_SHORT = FIREWORKS_DEFAULT_MODEL.split("/")[-1]
