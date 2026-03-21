"""Test per-user schema management."""
import pytest

def test_schema_name_from_email():
    from src.database.schema_manager import schema_name_for_user
    assert schema_name_for_user("james@example.com") == "user_james"
    assert schema_name_for_user("alice.bob@example.com") == "user_alice_bob"

def test_schema_name_sanitizes_special_chars():
    from src.database.schema_manager import schema_name_for_user
    # Should only allow alphanumeric and underscore
    assert schema_name_for_user("test+user@example.com") == "user_test_user"
    assert schema_name_for_user("a'b@evil.com") == "user_a_b"

def test_schema_name_rejects_empty():
    from src.database.schema_manager import schema_name_for_user
    with pytest.raises(ValueError):
        schema_name_for_user("")
