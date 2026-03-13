"""
Core Conversation Pipeline

Clean, modular conversation handling for the companion.
Replaces the 12+ context source assembly with a focused 6-source pipeline.
"""

from .context_builder import ContextBuilder
from .pipeline import ConversationPipeline

__all__ = ['ContextBuilder', 'ConversationPipeline']
