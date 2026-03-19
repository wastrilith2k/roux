"""
Image Generation — provider abstraction for generating images.

Supports:
- NanoBanana (Gemini-based, free with Google AI Studio key) — default for text-to-image
- RunComfy (ComfyUI-based, paid) — for custom workflows, LoRA models, advanced pipelines

Provider selection per companion (in persona.yaml):
  image:
    provider: nanobanana          # or 'runcomfy'
    runcomfy_deployment_id: "..." # if set, RunComfy is used regardless of provider setting
    content_rules: gemini         # 'gemini' (NanoBanana) or 'unrestricted' (RunComfy)

Content rules:
  gemini      — Google's Gemini policies: no NSFW, limited violence, no real people likenesses
  unrestricted — RunComfy with custom workflows: depends on your LoRA and pipeline
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Content rule descriptions — injected into image generation prompts
CONTENT_RULES = {
    'gemini': (
        "Content policy: No explicit/NSFW content. No graphic violence. "
        "No real person likenesses. Keep imagery safe and creative."
    ),
    'unrestricted': (
        "No content restrictions from the image provider. "
        "Follow the companion's personality and relationship context for appropriate content."
    ),
}


def get_content_rules(rule_name: str) -> str:
    """Get content rule description for prompt injection."""
    return CONTENT_RULES.get(rule_name, CONTENT_RULES['gemini'])


def get_image_provider_for_companion(companion_id: str = None):
    """Get the image provider for a specific companion.

    Logic:
    1. If the companion's persona.yaml has a runcomfy_deployment_id → RunComfy
    2. If the companion's persona.yaml sets provider: runcomfy → RunComfy
    3. Otherwise → NanoBanana (free default)
    4. Falls back to whatever is available if the preferred provider isn't configured

    Args:
        companion_id: The companion to get the provider for. If None, uses env default.

    Returns:
        (provider, content_rules_text) tuple, or (None, '') if nothing available
    """
    preferred = os.environ.get('IMAGE_PROVIDER', 'nanobanana').lower()
    content_rules = 'gemini'
    has_deployment_id = False

    # Check companion-specific config
    if companion_id:
        try:
            from src.config.persona_config import get_persona_config
            config = get_persona_config(companion_id=companion_id)
            raw = config._raw_config if hasattr(config, '_raw_config') else {}
            image_config = raw.get('image', {})
            preferred = image_config.get('provider', preferred)
            content_rules = image_config.get('content_rules', content_rules)
            deployment_id = image_config.get('runcomfy_deployment_id', '')
            if deployment_id:
                has_deployment_id = True
                preferred = 'runcomfy'
        except Exception:
            pass

    rules_text = get_content_rules(content_rules)

    # If companion has a RunComfy deployment ID, use RunComfy
    if has_deployment_id or preferred == 'runcomfy':
        try:
            from .runcomfy_manager import get_runcomfy_manager
            manager = get_runcomfy_manager()
            if manager.is_enabled():
                return manager, rules_text
        except ImportError:
            pass
        logger.debug("RunComfy requested but not available, falling back to NanoBanana")

    # Default: NanoBanana
    from .nanobanana_provider import get_nanobanana_provider
    provider = get_nanobanana_provider()
    if provider.is_enabled():
        return provider, get_content_rules('gemini')

    # Last resort: try RunComfy even if NanoBanana was preferred
    try:
        from .runcomfy_manager import get_runcomfy_manager
        manager = get_runcomfy_manager()
        if manager.is_enabled():
            return manager, get_content_rules('unrestricted')
    except ImportError:
        pass

    logger.warning("No image generation provider configured")
    return None, ''


def get_image_provider():
    """Get the default image provider (no companion context). For backwards compat."""
    provider, _ = get_image_provider_for_companion(None)
    return provider
