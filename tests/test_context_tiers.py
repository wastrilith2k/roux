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
