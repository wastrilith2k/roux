"""Verify tables.py exports all expected constants and they're plain strings."""


def test_all_table_constants_are_strings():
    from src.database import tables
    # String table-name constants are ALL_CAPS; lookup sets (PUBLIC_TABLES,
    # USER_SCHEMA_TABLES) are also ALL_CAPS but are sets — exclude them.
    all_caps = {k: v for k, v in vars(tables).items() if k.isupper() and not k.startswith('_')}
    string_constants = {k: v for k, v in all_caps.items() if isinstance(v, str)}
    assert len(string_constants) >= 45, f"Expected 45+ table constants, got {len(string_constants)}"
    for name, value in string_constants.items():
        assert isinstance(value, str), f"{name} should be str, got {type(value)}"
        assert len(value) > 0, f"{name} should not be empty"


def test_public_schema_tables_listed():
    from src.database.tables import PUBLIC_TABLES
    assert 'users' in PUBLIC_TABLES
    assert 'sessions' in PUBLIC_TABLES
    assert 'user_profiles' in PUBLIC_TABLES
    assert 'user_companions' in PUBLIC_TABLES


def test_user_schema_tables_listed():
    from src.database.tables import USER_SCHEMA_TABLES
    assert 'messages' in USER_SCHEMA_TABLES
    assert 'facts' in USER_SCHEMA_TABLES
    assert 'companion_goals' in USER_SCHEMA_TABLES
