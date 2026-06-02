import tempfile
from pathlib import Path
from src.memory.entity_profile_manager import EntityProfileManager


SAMPLE_MD = """\
---
name: TestEntity
age: 30
aliases:
  - tester
does_not_have:
  - No car.
---

# TestEntity — Profile

Test entity profile narrative.

## Personality
- Calm
- Methodical
"""


def _make_manager(md_content: str) -> EntityProfileManager:
    tmpdir = tempfile.mkdtemp()
    (Path(tmpdir) / "testentity.md").write_text(md_content)
    return EntityProfileManager(profiles_dir=Path(tmpdir))


def test_get_profile_returns_frontmatter_dict():
    manager = _make_manager(SAMPLE_MD)
    profile = manager.get_profile('testentity')
    assert profile is not None
    assert profile['name'] == 'TestEntity'
    assert profile['age'] == 30


def test_get_grounding_context_returns_markdown_body():
    manager = _make_manager(SAMPLE_MD)
    ctx = manager.get_grounding_context()
    assert 'Test entity profile narrative' in ctx
    assert 'Personality' in ctx


def test_fallback_to_yaml():
    """Existing YAML files must still load during migration period."""
    import yaml
    tmpdir = tempfile.mkdtemp()
    (Path(tmpdir) / "legacy.yaml").write_text("name: Legacy\nage: 25\n")
    manager = EntityProfileManager(profiles_dir=Path(tmpdir))
    profile = manager.get_profile('legacy')
    assert profile is not None
    assert profile['name'] == 'Legacy'


def test_returns_none_for_unknown_entity():
    manager = _make_manager(SAMPLE_MD)
    assert manager.get_profile('nobody') is None
