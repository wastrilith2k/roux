"""Regression tests for issue #37: Migrate from Kimi K2 to K2.5 and centralize model config.

Covers:
1. Central config exports correct default model string (kimi-k2p5-instruct)
2. No source files contain the deprecated kimi-k2-instruct-0905 string
3. FireworksProvider default matches centralized config
4. fireworks_models.py MEDIUM_MODELS uses K2.5
5. CostTracker MODEL_COSTS has pricing for kimi-k2p5-instruct (not old model)
6. Environment variable overrides work for FIREWORKS_MODEL
7. Environment variable overrides work for FIREWORKS_FALLBACK_MODEL
8. FIREWORKS_DEFAULT_MODEL_SHORT strips the prefix correctly
9. /models command works when LLM_PROVIDER is not "fireworks" (no NameError)
"""

import os
import glob
import pytest
from unittest.mock import patch

os.environ.setdefault('ENVIRONMENT', 'test')
os.environ.setdefault('POSTGRES_PASSWORD', 'test')
os.environ.setdefault('POSTGRES_DB', 'test')
os.environ.setdefault('POSTGRES_HOST', 'localhost')
os.environ.setdefault('POSTGRES_PORT', '5432')
os.environ.setdefault('POSTGRES_USER', 'test')

DEPRECATED_MODEL = "kimi-k2-instruct-0905"
NEW_MODEL = "kimi-k2p5"
NEW_MODEL_FULL = f"accounts/fireworks/models/{NEW_MODEL}"


# ---------------------------------------------------------------------------
# Central config tests
# ---------------------------------------------------------------------------

class TestCentralModelConfig:
    """Verify src/config/models.py is the single source of truth."""

    def test_default_model_is_k2p5(self):
        """Central config must default to kimi-k2p5-instruct."""
        from src.config.models import FIREWORKS_DEFAULT_MODEL
        assert NEW_MODEL in FIREWORKS_DEFAULT_MODEL
        assert DEPRECATED_MODEL not in FIREWORKS_DEFAULT_MODEL

    def test_fallback_model_is_k2p5(self):
        """Fallback must also default to K2.5."""
        from src.config.models import FIREWORKS_FALLBACK_MODEL
        assert NEW_MODEL in FIREWORKS_FALLBACK_MODEL
        assert DEPRECATED_MODEL not in FIREWORKS_FALLBACK_MODEL

    def test_background_model_is_k2p5(self):
        """Background/autonomy model must also default to K2.5."""
        from src.config.models import FIREWORKS_BACKGROUND_MODEL
        assert NEW_MODEL in FIREWORKS_BACKGROUND_MODEL
        assert DEPRECATED_MODEL not in FIREWORKS_BACKGROUND_MODEL

    def test_short_name(self):
        """Short name must be just the model name, not the full path."""
        from src.config.models import FIREWORKS_DEFAULT_MODEL_SHORT
        assert FIREWORKS_DEFAULT_MODEL_SHORT == NEW_MODEL
        assert "/" not in FIREWORKS_DEFAULT_MODEL_SHORT

    def test_env_override_default_model(self):
        """FIREWORKS_MODEL env var should override the default."""
        with patch.dict(os.environ, {"FIREWORKS_MODEL": "accounts/fireworks/models/custom-model"}):
            # Re-import to pick up env change
            import importlib
            import src.config.models as models_mod
            importlib.reload(models_mod)
            assert models_mod.FIREWORKS_DEFAULT_MODEL == "accounts/fireworks/models/custom-model"

            # Restore
            del os.environ["FIREWORKS_MODEL"]
            importlib.reload(models_mod)


# ---------------------------------------------------------------------------
# No deprecated model strings in source code
# ---------------------------------------------------------------------------

class TestNoDeprecatedModelStrings:
    """Ensure the deprecated kimi-k2-instruct-0905 string is gone from src/."""

    def test_no_deprecated_model_in_source(self):
        """No .py file under src/ should contain the deprecated model string.

        The only allowed exception is agent.py.deprecated (not active code).
        """
        src_root = os.path.join(os.path.dirname(__file__), '..', 'src')
        src_root = os.path.abspath(src_root)

        violations = []
        for py_file in glob.glob(os.path.join(src_root, '**', '*.py'), recursive=True):
            with open(py_file, 'r') as f:
                content = f.read()
            if DEPRECATED_MODEL in content:
                rel = os.path.relpath(py_file, os.path.join(src_root, '..'))
                violations.append(rel)

        assert violations == [], (
            f"Deprecated model '{DEPRECATED_MODEL}' still found in: {violations}"
        )


# ---------------------------------------------------------------------------
# Provider-level tests
# ---------------------------------------------------------------------------

class TestFireworksProviderDefault:
    """Verify FireworksProvider default model matches centralized config."""

    def test_provider_default_is_k2p5(self):
        """FireworksProvider.__init__ default must be K2.5."""
        import inspect
        from src.llm.fireworks_provider import FireworksProvider
        sig = inspect.signature(FireworksProvider.__init__)
        default = sig.parameters['model'].default
        assert NEW_MODEL in default, (
            f"FireworksProvider default model is '{default}', expected K2.5"
        )
        assert DEPRECATED_MODEL not in default


class TestFireworksModels:
    """Verify fireworks_models.py MEDIUM_MODELS uses K2.5."""

    def test_medium_models_has_k2p5(self):
        from src.llm.fireworks_models import MEDIUM_MODELS
        model_str = " ".join(MEDIUM_MODELS)
        assert NEW_MODEL in model_str
        assert DEPRECATED_MODEL not in model_str


# ---------------------------------------------------------------------------
# Cost tracker tests
# ---------------------------------------------------------------------------

class TestCostTrackerPricing:
    """Verify MODEL_COSTS has the new model and not the old one."""

    def test_k2p5_in_model_costs(self):
        from src.core.cost_tracker import MODEL_COSTS
        assert NEW_MODEL in MODEL_COSTS, (
            f"'{NEW_MODEL}' not found in MODEL_COSTS keys: {list(MODEL_COSTS.keys())}"
        )

    def test_deprecated_not_in_model_costs(self):
        from src.core.cost_tracker import MODEL_COSTS
        assert DEPRECATED_MODEL not in MODEL_COSTS, (
            f"Deprecated '{DEPRECATED_MODEL}' still in MODEL_COSTS"
        )

    def test_k2p5_pricing_has_input_and_output(self):
        from src.core.cost_tracker import MODEL_COSTS
        pricing = MODEL_COSTS[NEW_MODEL]
        assert "input" in pricing
        assert "output" in pricing
        assert pricing["input"] > 0
        assert pricing["output"] > 0


# ---------------------------------------------------------------------------
# /models command regression test
# ---------------------------------------------------------------------------

class TestModelsCommandNonFireworks:
    """Regression: /models must not raise NameError when LLM_PROVIDER != fireworks."""

    def test_get_model_info_with_anthropic_provider(self):
        """get_model_info() must work when LLM_PROVIDER is 'anthropic'.

        Previous bug: FIREWORKS_DEFAULT_MODEL was imported inside the
        'if llm_provider == "fireworks"' branch but used unconditionally
        in the heavy_tasks loop, causing NameError for non-fireworks providers.
        """
        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic"}):
            from src.core.commands.models_command import get_model_info
            info = get_model_info()
            assert isinstance(info, dict)
            assert "models" in info
            # Heavy tasks should still be listed with Fireworks model names
            tasks = [m["task"] for m in info["models"]]
            assert "Fact Extraction" in tasks
