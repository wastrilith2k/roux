import os
import importlib


def test_observational_memory_enabled_by_default():
    """OBSERVATIONAL_MEMORY_ENABLED must be True when env var is not set."""
    # Remove env var if set, reload module, check default
    original = os.environ.pop('OBSERVATIONAL_MEMORY_ENABLED', None)
    try:
        import src.memory.observational_memory as om
        importlib.reload(om)
        assert om.OBSERVATIONAL_MEMORY_ENABLED is True, (
            "OBSERVATIONAL_MEMORY_ENABLED should default to True"
        )
    finally:
        if original is not None:
            os.environ['OBSERVATIONAL_MEMORY_ENABLED'] = original
        import src.memory.observational_memory as om
        importlib.reload(om)


def test_observational_memory_can_be_disabled_via_env():
    """Should be possible to disable via env var."""
    original = os.environ.get('OBSERVATIONAL_MEMORY_ENABLED')
    os.environ['OBSERVATIONAL_MEMORY_ENABLED'] = 'false'
    try:
        import src.memory.observational_memory as om
        importlib.reload(om)
        assert om.OBSERVATIONAL_MEMORY_ENABLED is False
    finally:
        if original is not None:
            os.environ['OBSERVATIONAL_MEMORY_ENABLED'] = original
        else:
            os.environ.pop('OBSERVATIONAL_MEMORY_ENABLED', None)
        import src.memory.observational_memory as om
        importlib.reload(om)
