"""
LLM Provider Abstraction Layer

Clean interface for multiple LLM providers (Fireworks AI, Anthropic Claude).
Replaces hardcoded provider selection scattered across codebase.
"""

from .provider_factory import get_llm_provider
from .provider_interface import LLMProvider

__all__ = ['get_llm_provider', 'LLMProvider']
