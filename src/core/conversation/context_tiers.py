from enum import IntEnum


class ContextTier(IntEnum):
    """CoALA memory layer — lower value = higher priority, never dropped first."""
    PROCEDURAL = 1       # How to behave — never evicted
    PERMANENT = 2        # Core identity facts — never evicted
    DURABLE = 3          # Relationship state, scene — survive most pressure
    DISTILLED_EPISODIC = 4  # Synthesized biographies/events — prefer over raw
    RAW_EPISODIC = 5     # Raw episode matches — fallback
    EPHEMERAL = 6        # Time-of-day guesses, simulated activities — drop first


# Map ConversationContext field names to their CoALA tier.
SECTION_TIERS: dict[str, ContextTier] = {
    # Procedural — how to behave
    'personality':            ContextTier.PERMANENT,
    'presence_mode':          ContextTier.PROCEDURAL,
    'derived_scene_context':  ContextTier.PROCEDURAL,

    # Permanent semantic — core identity, never wrong to have
    'entity_profiles':        ContextTier.PERMANENT,
    'core_memory':            ContextTier.PERMANENT,
    'relationship_dynamics':  ContextTier.PERMANENT,
    'relationship_evaluation': ContextTier.PERMANENT,

    # Durable semantic — valuable but survives pressure
    'memories':               ContextTier.DURABLE,
    'graphiti_context':       ContextTier.DURABLE,
    'relationship_insights':  ContextTier.DURABLE,
    'scene_state':            ContextTier.DURABLE,
    'internal_state':         ContextTier.DURABLE,
    'session_summary':        ContextTier.DURABLE,
    'temporal_context':       ContextTier.DURABLE,
    'continuity_context':     ContextTier.DURABLE,
    'location':               ContextTier.DURABLE,

    # Distilled episodic — synthesized history, prefer over raw
    'biographies':            ContextTier.DISTILLED_EPISODIC,
    'synthesized_events':     ContextTier.DISTILLED_EPISODIC,
    'observations_context':   ContextTier.DISTILLED_EPISODIC,

    # Raw episodic — matched episodes, fallback
    'episode_context':        ContextTier.RAW_EPISODIC,
    'conversation_history':   ContextTier.RAW_EPISODIC,
    'conversation_turns':     ContextTier.RAW_EPISODIC,

    # Ephemeral — simulated/inferred state, drop first
    'reflections_context':    ContextTier.EPHEMERAL,
    'opinions_context':       ContextTier.EPHEMERAL,
    'curiosity_context':      ContextTier.EPHEMERAL,
    'goals_context':          ContextTier.EPHEMERAL,
    'values_context':         ContextTier.EPHEMERAL,
    'activities_context':     ContextTier.EPHEMERAL,
    'fertility_context':      ContextTier.EPHEMERAL,
    'user_context':           ContextTier.EPHEMERAL,
    'schedule':               ContextTier.EPHEMERAL,
}
