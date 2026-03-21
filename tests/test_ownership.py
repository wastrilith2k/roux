# tests/test_ownership.py
"""Test ownership module imports and function signatures."""


def test_ownership_module_imports():
    from src.database.ownership import (
        user_owns_companion,
        assign_companion,
        get_user_companions,
        revoke_companion,
    )
    # Verify they're callable
    assert callable(user_owns_companion)
    assert callable(assign_companion)
    assert callable(get_user_companions)
    assert callable(revoke_companion)
