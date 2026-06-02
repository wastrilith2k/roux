from src.core.conversation.context_tiers import ContextTier, SECTION_TIERS

def test_permanent_sections_never_evicted():
    permanent = [k for k, v in SECTION_TIERS.items() if v == ContextTier.PERMANENT]
    assert 'core_memory' in permanent
    assert 'entity_profiles' in permanent
    assert 'personality' in permanent

def test_procedural_sections_never_evicted():
    procedural = [k for k, v in SECTION_TIERS.items() if v == ContextTier.PROCEDURAL]
    assert 'presence_mode' in procedural
    assert 'derived_scene_context' in procedural

def test_ephemeral_sections_dropped_first():
    ephemeral = [k for k, v in SECTION_TIERS.items() if v == ContextTier.EPHEMERAL]
    assert 'user_context' in ephemeral
    assert 'activities_context' in ephemeral

def test_distilled_episodic_above_raw_episodic():
    tiers = SECTION_TIERS
    assert tiers['biographies'].value < tiers['episode_context'].value
    assert tiers['synthesized_events'].value < tiers['episode_context'].value

def test_permanent_and_procedural_have_lowest_values():
    non_evictable_tiers = {ContextTier.PROCEDURAL, ContextTier.PERMANENT}
    for name, tier in SECTION_TIERS.items():
        if tier in non_evictable_tiers:
            assert tier.value <= 2, f"{name} should have tier <= 2 to be protected"

def test_distilled_lower_than_raw():
    assert SECTION_TIERS['biographies'] < SECTION_TIERS['episode_context']
    assert SECTION_TIERS['synthesized_events'] < SECTION_TIERS['episode_context']

def test_section_tiers_covers_all_prompt_sections():
    """All ConversationContext string fields used as prompt sections must have a tier."""
    # These are the known prompt-injectable string fields (not metadata or list fields)
    required_sections = {
        'entity_profiles', 'personality', 'memories', 'biographies',
        'conversation_history', 'continuity_context', 'relationship_insights',
        'schedule', 'location', 'scene_state', 'internal_state', 'user_context',
        'values_context', 'activities_context', 'temporal_context',
        'graphiti_context', 'synthesized_events', 'episode_context',
        'core_memory', 'reflections_context', 'opinions_context',
        'curiosity_context', 'goals_context', 'fertility_context',
        'observations_context', 'session_summary', 'relationship_dynamics',
        'relationship_evaluation', 'presence_mode', 'derived_scene_context',
    }
    missing = required_sections - set(SECTION_TIERS.keys())
    assert not missing, f"Fields missing from SECTION_TIERS: {missing}"
