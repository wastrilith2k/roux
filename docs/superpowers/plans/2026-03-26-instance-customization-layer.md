# Instance Customization Layer — Eliminate Fork Conflicts

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate all hardcoded companion/instance-specific values from the codebase so that companion instances are deployments of Roux (driven by config), not forks.

**Architecture:** Replace every hardcoded companion name (`'kai'`, `'mira'`), user identity (`'admin'`), and instance-specific default with references to environment variables (`COMPANION_ID`) or the existing `get_persona_config()` / `companions.yaml` configuration. The codebase already has the right patterns (`os.environ.get('COMPANION_ID', 'default')` and `get_persona_config()`); this work extends those patterns to the ~15 locations that still use hardcoded values.

**Tech Stack:** Python 3.11, pytest, YAML config, environment variables

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `src/routes/observe_routes.py` | Replace 7x hardcoded `'kai'` defaults + hardcoded peer mapping |
| Modify | `src/database/cost_tracking_schema.sql` | Replace `'ESME AI'` header and `'admin'` seed user |
| Modify | `src/llm/openrouter_provider.py` | Replace hardcoded GitHub URL default |
| Modify | `src/utils/error_tracker.py` | Replace `'kai'` in docstring example |
| Modify | `scripts/simulate_relationship.py` | Replace hardcoded `['kai', 'mira']` default |
| Modify | `scripts/generate_report.py` | Replace hardcoded `['kai', 'mira']` default |
| Create | `src/config/companion_registry.py` | Shared utility to load companions.yaml + get enabled companion IDs |
| Create | `tests/test_instance_customization.py` | Regression tests for all hardcoded value elimination |

---

### Task 1: Create shared companion registry utility

The scripts `seed_database.py`, `admin.py`, `simulate_relationship.py`, and `generate_report.py` all need to read companion IDs from `companions.yaml`. Currently `seed_database.py` has its own `load_companions_registry()`, and the other scripts hardcode `['kai', 'mira']`. Create a shared utility.

**Files:**
- Create: `src/config/companion_registry.py`
- Test: `tests/test_instance_customization.py`

- [ ] **Step 1: Write the failing test for `get_enabled_companion_ids()`**

```python
"""Regression tests for issue #49 — instance customization layer.

Ensures no hardcoded companion names, user identities, or instance-specific
defaults remain in the codebase. Every identity value must come from
configuration (env vars, persona YAML, or companions.yaml).
"""
import os
import pytest
from unittest.mock import patch, mock_open

from src.config.companion_registry import get_enabled_companion_ids


class TestCompanionRegistry:
    """Test the shared companion registry utility."""

    def test_get_enabled_companion_ids_returns_list(self):
        """get_enabled_companion_ids() should return a list of enabled companion IDs."""
        ids = get_enabled_companion_ids()
        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_get_enabled_companion_ids_filters_disabled(self):
        """Only companions with enabled: true should be returned."""
        yaml_content = """
companions:
  alpha:
    persona_config: instances/alpha/persona.yaml
    enabled: true
  beta:
    persona_config: instances/beta/persona.yaml
    enabled: false
"""
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            with patch('src.config.companion_registry._find_project_root') as mock_root:
                from pathlib import Path
                mock_root.return_value = Path('/fake')
                with patch('pathlib.Path.exists', return_value=True):
                    ids = get_enabled_companion_ids()
                    assert 'alpha' in ids
                    assert 'beta' not in ids

    def test_get_enabled_companion_ids_no_hardcoded_names(self):
        """Return value must not contain hardcoded defaults like 'kai' or 'mira'
        when companions.yaml has different content."""
        yaml_content = """
companions:
  custom_bot:
    persona_config: instances/custom_bot/persona.yaml
    enabled: true
"""
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            with patch('src.config.companion_registry._find_project_root') as mock_root:
                from pathlib import Path
                mock_root.return_value = Path('/fake')
                with patch('pathlib.Path.exists', return_value=True):
                    ids = get_enabled_companion_ids()
                    assert ids == ['custom_bot']
                    assert 'kai' not in ids
                    assert 'mira' not in ids
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_instance_customization.py::TestCompanionRegistry::test_get_enabled_companion_ids_returns_list -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.config.companion_registry'`

- [ ] **Step 3: Write the companion registry module**

```python
"""
Companion Registry — Shared utility for loading companion metadata.

WHAT: Reads data/companions.yaml and exposes helpers for listing enabled
      companion IDs. Replaces ad-hoc YAML loading scattered across scripts.
WHY:  Scripts that need companion IDs were either hardcoding ['kai', 'mira']
      or duplicating YAML loading logic. This centralizes it.
HOW:  Reads companions.yaml from the project root, filters by enabled flag.
"""

import logging
from pathlib import Path
from typing import List

import yaml

logger = logging.getLogger(__name__)


def _find_project_root() -> Path:
    """Locate the project root by looking for a data/ directory."""
    candidates = [
        Path('/app'),
        Path(__file__).parent.parent.parent,
    ]
    for path in candidates:
        try:
            if (path / 'data').exists():
                return path
        except PermissionError:
            continue
    return Path('/app')


def get_enabled_companion_ids() -> List[str]:
    """Return list of enabled companion IDs from companions.yaml.

    Falls back to empty list if the file doesn't exist or can't be parsed.
    """
    project_root = _find_project_root()
    registry_path = project_root / 'data' / 'companions.yaml'

    if not registry_path.exists():
        logger.warning(f"No companions.yaml at {registry_path}")
        return []

    try:
        with open(registry_path) as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Failed to load companions.yaml: {e}")
        return []

    companions = data.get('companions', {})
    return [
        cid for cid, cfg in companions.items()
        if cfg.get('enabled', False)
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_instance_customization.py::TestCompanionRegistry -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/config/companion_registry.py tests/test_instance_customization.py
git commit -m "feat: add shared companion registry utility (issue #49)"
```

---

### Task 2: Replace hardcoded `'kai'` defaults in observe_routes.py

The observe routes have 7 endpoints that default `companion_id` to `'kai'`, plus a hardcoded peer mapping `{'kai': 'mira', 'mira': 'kai'}`. These should use `COMPANION_ID` env var.

**Files:**
- Modify: `src/routes/observe_routes.py`
- Test: `tests/test_instance_customization.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_instance_customization.py`:

```python
class TestObserveRoutesNoHardcodedDefaults:
    """observe_routes.py must not hardcode companion names."""

    def test_no_hardcoded_kai_default_in_observe_routes(self):
        """The string 'kai' must not appear as a default value in observe_routes.py."""
        import ast
        from pathlib import Path

        source_path = Path(__file__).parent.parent / 'src' / 'routes' / 'observe_routes.py'
        source = source_path.read_text()

        # Parse the AST and look for request.args.get('companion_id', 'kai')
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == 'get' and len(node.args) >= 2:
                    # Check if first arg is 'companion_id' and second is 'kai'
                    if (isinstance(node.args[0], ast.Constant) and node.args[0].value == 'companion_id'
                            and isinstance(node.args[1], ast.Constant) and node.args[1].value == 'kai'):
                        pytest.fail(
                            f"Line {node.lineno}: hardcoded default 'kai' found in "
                            f"request.args.get('companion_id', 'kai'). "
                            f"Should use COMPANION_ID env var."
                        )

    def test_no_hardcoded_peer_mapping_in_observe_routes(self):
        """The peer mapping dict must not hardcode specific companion names."""
        from pathlib import Path

        source_path = Path(__file__).parent.parent / 'src' / 'routes' / 'observe_routes.py'
        source = source_path.read_text()

        # The old hardcoded mapping was: {'kai': 'mira', 'mira': 'kai'}
        assert "'kai': 'mira'" not in source, (
            "Hardcoded peer mapping {'kai': 'mira', ...} found in observe_routes.py"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_instance_customization.py::TestObserveRoutesNoHardcodedDefaults -v`
Expected: FAIL — both tests should fail because observe_routes.py still has hardcoded values

- [ ] **Step 3: Fix observe_routes.py — replace defaults and peer mapping**

Replace the default companion_id pattern. Change every occurrence of:
```python
companion_id = request.args.get('companion_id', 'kai')
```
to:
```python
companion_id = request.args.get('companion_id') or os.environ.get('COMPANION_ID', 'default')
```

Replace the `_companion_email` function. Change:
```python
def _companion_email(companion_id: str) -> str:
    """Map companion_id to the email used in user_state/messages.

    In the simulation, Kai's state is stored with email='mira@companion.local'
    and companion_id='kai'. For querying messages/facts/opinions, we filter
    by companion_id directly.
    """
    other = {'kai': 'mira', 'mira': 'kai'}
    peer = other.get(companion_id, companion_id)
    return f"{peer}@companion.local"
```
to:
```python
def _companion_email(companion_id: str) -> str:
    """Map companion_id to the peer email used in user_state/messages.

    In simulation mode, each companion's state is stored under its peer's
    email (e.g., companion A's state uses B's email). The peer mapping is
    loaded from companions.yaml. For single-companion mode, falls back to
    the companion_id itself.
    """
    from src.config.companion_registry import get_enabled_companion_ids
    enabled = get_enabled_companion_ids()
    if len(enabled) == 2 and companion_id in enabled:
        peer = [cid for cid in enabled if cid != companion_id][0]
    else:
        peer = companion_id
    return f"{peer}@companion.local"
```

Update the module docstring. Change:
```python
WHY:  The simulation runner emits events as Kai and Mira talk. This module
      bridges those events to the browser so you can watch in real-time.
```
to:
```python
WHY:  The simulation runner emits events as companions talk. This module
      bridges those events to the browser so you can watch in real-time.
```

Also update the `_companion_email` comment on line 38:
```python
    In the simulation, Kai's state is stored with email='mira@companion.local'
```
to:
```python
    In simulation mode, each companion's state is stored under its peer's
```
(already covered in the replacement above)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_instance_customization.py::TestObserveRoutesNoHardcodedDefaults -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/routes/observe_routes.py tests/test_instance_customization.py
git commit -m "fix: remove hardcoded 'kai' defaults from observe routes (issue #49)"
```

---

### Task 3: Replace hardcoded values in cost_tracking_schema.sql

The SQL file has `'ESME AI'` in the header comment and `'admin'` as a hardcoded seed user.

**Files:**
- Modify: `src/database/cost_tracking_schema.sql`
- Test: `tests/test_instance_customization.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_instance_customization.py`:

```python
class TestSchemaNoHardcodedIdentity:
    """SQL schema files must not hardcode instance-specific identity."""

    def test_no_esme_in_cost_tracking_schema(self):
        """The string 'ESME' must not appear in cost_tracking_schema.sql."""
        from pathlib import Path
        schema_path = Path(__file__).parent.parent / 'src' / 'database' / 'cost_tracking_schema.sql'
        content = schema_path.read_text()
        assert 'ESME' not in content.upper() or 'ESME' not in content, (
            "Hardcoded 'ESME AI' found in cost_tracking_schema.sql — "
            "should use generic 'Companion Framework' name"
        )

    def test_no_hardcoded_admin_seed_user(self):
        """The admin seed user must not be hardcoded in the schema."""
        from pathlib import Path
        schema_path = Path(__file__).parent.parent / 'src' / 'database' / 'cost_tracking_schema.sql'
        content = schema_path.read_text()
        assert "VALUES ('admin')" not in content, (
            "Hardcoded 'admin' seed user found in cost_tracking_schema.sql — "
            "should use a generic default like 'default'"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_instance_customization.py::TestSchemaNoHardcodedIdentity -v`
Expected: FAIL — both assertions fail

- [ ] **Step 3: Fix cost_tracking_schema.sql**

Change line 2 from:
```sql
-- ESME AI - UNIFIED COST TRACKING DATABASE SCHEMA
```
to:
```sql
-- COMPANION FRAMEWORK - UNIFIED COST TRACKING DATABASE SCHEMA
```

Change lines 297-298 from:
```sql
-- Create default budget for admin user
INSERT OR IGNORE INTO budget_limits (user_id) VALUES ('admin');
```
to:
```sql
-- Create default budget limit
INSERT OR IGNORE INTO budget_limits (user_id) VALUES ('default');
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_instance_customization.py::TestSchemaNoHardcodedIdentity -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/database/cost_tracking_schema.sql tests/test_instance_customization.py
git commit -m "fix: remove hardcoded 'ESME AI' and 'admin' from cost tracking schema (issue #49)"
```

---

### Task 4: Replace hardcoded GitHub URL in openrouter_provider.py

The OpenRouter provider has a hardcoded GitHub URL with the repository owner's username. It's already env-var-overridable, but the default leaks instance identity.

**Files:**
- Modify: `src/llm/openrouter_provider.py`
- Test: `tests/test_instance_customization.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_instance_customization.py`:

```python
class TestOpenRouterNoHardcodedDefaults:
    """OpenRouter provider must not hardcode instance-specific URLs."""

    def test_no_hardcoded_github_username_in_referer(self):
        """The default OPENROUTER_REFERER must not contain a hardcoded GitHub username."""
        from pathlib import Path
        source_path = Path(__file__).parent.parent / 'src' / 'llm' / 'openrouter_provider.py'
        source = source_path.read_text()
        assert 'wastrilith2k' not in source, (
            "Hardcoded GitHub username 'wastrilith2k' found in openrouter_provider.py — "
            "should use a generic project URL or empty default"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_instance_customization.py::TestOpenRouterNoHardcodedDefaults -v`
Expected: FAIL

- [ ] **Step 3: Fix openrouter_provider.py**

Change:
```python
                "HTTP-Referer": os.environ.get(
                    "OPENROUTER_REFERER", "https://github.com/wastrilith2k/roux"
                ),
```
to:
```python
                "HTTP-Referer": os.environ.get(
                    "OPENROUTER_REFERER", "https://github.com/companion-framework/roux"
                ),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_instance_customization.py::TestOpenRouterNoHardcodedDefaults -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm/openrouter_provider.py tests/test_instance_customization.py
git commit -m "fix: remove hardcoded GitHub username from OpenRouter referer default (issue #49)"
```

---

### Task 5: Replace hardcoded defaults in CLI scripts

Both `simulate_relationship.py` and `generate_report.py` hardcode `['kai', 'mira']` as default companion IDs. These should read from companions.yaml.

**Files:**
- Modify: `scripts/simulate_relationship.py`
- Modify: `scripts/generate_report.py`
- Test: `tests/test_instance_customization.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_instance_customization.py`:

```python
class TestScriptsNoHardcodedDefaults:
    """CLI scripts must not hardcode companion IDs."""

    def test_no_hardcoded_kai_mira_in_simulate_relationship(self):
        """simulate_relationship.py must not hardcode ['kai', 'mira'] as defaults."""
        from pathlib import Path
        source_path = Path(__file__).parent.parent / 'scripts' / 'simulate_relationship.py'
        source = source_path.read_text()
        assert "default=['kai', 'mira']" not in source, (
            "Hardcoded default=['kai', 'mira'] found in simulate_relationship.py"
        )

    def test_no_hardcoded_kai_mira_in_generate_report(self):
        """generate_report.py must not hardcode ['kai', 'mira'] as defaults."""
        from pathlib import Path
        source_path = Path(__file__).parent.parent / 'scripts' / 'generate_report.py'
        source = source_path.read_text()
        assert "default=['kai', 'mira']" not in source, (
            "Hardcoded default=['kai', 'mira'] found in generate_report.py"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_instance_customization.py::TestScriptsNoHardcodedDefaults -v`
Expected: FAIL

- [ ] **Step 3: Fix simulate_relationship.py**

Change:
```python
    parser.add_argument('--companions', nargs='+', default=['kai', 'mira'],
                        help='Companion IDs to simulate (default: kai mira)')
```
to:
```python
    parser.add_argument('--companions', nargs='+', default=None,
                        help='Companion IDs to simulate (default: reads from companions.yaml)')
```

Then after parsing, add the default resolution (after the `args = parser.parse_args()` line):

```python
    if args.companions is None:
        from src.config.companion_registry import get_enabled_companion_ids
        args.companions = get_enabled_companion_ids()
        if not args.companions:
            parser.error("No enabled companions found in companions.yaml. Use --companions to specify explicitly.")
```

- [ ] **Step 4: Fix generate_report.py**

Change:
```python
    parser.add_argument('--companions', nargs='+', default=['kai', 'mira'], help='Companion IDs')
```
to:
```python
    parser.add_argument('--companions', nargs='+', default=None,
                        help='Companion IDs (default: reads from companions.yaml)')
```

Then after parsing, add the default resolution (after the `args = parser.parse_args()` line):

```python
    if args.companions is None:
        from src.config.companion_registry import get_enabled_companion_ids
        args.companions = get_enabled_companion_ids()
        if not args.companions:
            parser.error("No enabled companions found in companions.yaml. Use --companions to specify explicitly.")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_instance_customization.py::TestScriptsNoHardcodedDefaults -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/simulate_relationship.py scripts/generate_report.py tests/test_instance_customization.py
git commit -m "fix: replace hardcoded companion ID defaults in CLI scripts with companions.yaml lookup (issue #49)"
```

---

### Task 6: Fix hardcoded companion name in error_tracker.py docstring

**Files:**
- Modify: `src/utils/error_tracker.py`
- Test: `tests/test_instance_customization.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_instance_customization.py`:

```python
class TestDocstringsNoHardcodedNames:
    """Docstrings and comments in source code must use generic names."""

    def test_no_kai_in_error_tracker_docstring(self):
        """error_tracker.py must not use 'kai' as an example companion_id."""
        from pathlib import Path
        source_path = Path(__file__).parent.parent / 'src' / 'utils' / 'error_tracker.py'
        source = source_path.read_text()
        assert "companion_id='kai'" not in source, (
            "Hardcoded companion_id='kai' found in error_tracker.py docstring"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_instance_customization.py::TestDocstringsNoHardcodedNames -v`
Expected: FAIL

- [ ] **Step 3: Fix error_tracker.py**

Change line 20:
```python
        record_error(e, module='fact_store', companion_id='kai')
```
to:
```python
        record_error(e, module='fact_store', companion_id='my_companion')
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_instance_customization.py::TestDocstringsNoHardcodedNames -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/utils/error_tracker.py tests/test_instance_customization.py
git commit -m "fix: remove hardcoded 'kai' from error_tracker docstring example (issue #49)"
```

---

### Task 7: Run full test suite and final commit

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 2: If any existing tests fail, investigate and fix**

Existing tests should not reference the removed hardcoded values. If they do, update them to use config-driven values.

- [ ] **Step 3: Final commit if any additional fixes were needed**

```bash
git add -A
git commit -m "fix: resolve issue #49 — instance customization layer, eliminate hardcoded fork conflicts"
```
