from src.core.personality_evolution import EvolvingPersonality


def _make_personality() -> EvolvingPersonality:
    """Create an EvolvingPersonality without DB connection."""
    p = EvolvingPersonality.__new__(EvolvingPersonality)
    p.user_id = 'test_user'
    p.dimensions = {
        'playfulness': 0.5, 'directness': 0.5, 'vulnerability': 0.3,
        'philosophical_tendency': 0.3, 'challenge_frequency': 0.3,
        'supportive_vs_tough_love': 0.5, 'formality': 0.3,
        'emotional_expressiveness': 0.5, 'intellectual_curiosity': 0.5,
        'spontaneity': 0.5,
    }
    from collections import deque
    p.interaction_feedback = deque(maxlen=50)
    p.evolution_history = []
    p._DRIFT_THRESHOLD = 0.15
    p._DRIFT_WINDOW_DAYS = 7
    return p


def test_check_drift_returns_empty_when_no_excess():
    p = _make_personality()
    history = [
        {'dimension': 'playfulness', 'delta': 0.03, 'days_ago': 1},
        {'dimension': 'playfulness', 'delta': 0.02, 'days_ago': 2},
    ]
    drift = p.check_drift(recent_history=history, threshold=0.15)
    assert drift == {}


def test_check_drift_detects_excess():
    p = _make_personality()
    # 6 updates of 0.03 = 0.18 > 0.15
    history = [
        {'dimension': 'playfulness', 'delta': 0.03, 'days_ago': i}
        for i in range(1, 7)
    ]
    drift = p.check_drift(recent_history=history, threshold=0.15)
    assert 'playfulness' in drift
    assert drift['playfulness'] > 0.15


def test_check_drift_ignores_entries_outside_window():
    p = _make_personality()
    history = [
        {'dimension': 'playfulness', 'delta': 0.05, 'days_ago': 10},  # outside 7-day window
        {'dimension': 'playfulness', 'delta': 0.02, 'days_ago': 1},   # inside
    ]
    drift = p.check_drift(recent_history=history, window_days=7, threshold=0.15)
    assert 'playfulness' not in drift  # only 0.02 inside window, below threshold
