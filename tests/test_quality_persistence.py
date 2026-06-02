import inspect
from src.database.db import CompanionDB


def test_store_message_accepts_quality_score():
    """store_message must accept a quality_score parameter."""
    sig = inspect.signature(CompanionDB.store_message)
    assert 'quality_score' in sig.parameters, (
        "store_message must have a quality_score parameter"
    )


def test_quality_score_parameter_defaults_to_none():
    sig = inspect.signature(CompanionDB.store_message)
    default = sig.parameters['quality_score'].default
    assert default is None, f"quality_score must default to None, got {default}"
