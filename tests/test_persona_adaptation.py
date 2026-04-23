"""Tests for two-level persona adaptation."""

import pytest
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from src.core.clock import SimulationClock, set_clock, SystemClock, PST
from src.autonomy.persona_adaptation import (
    detect_trait_signals,
    MicroAdaptation,
    store_pending_adaptation,
    get_pending_adaptations,
    clear_pending_adaptations,
    format_adaptations_for_value_inference,
    TRAIT_INDICATORS,
)

NOW = datetime(2026, 4, 1, 12, 0, tzinfo=PST)


@pytest.fixture(autouse=True)
def fixed_clock():
    set_clock(SimulationClock(start=NOW))
    yield
    set_clock(SystemClock())


class TestTraitDetection:
    """Test the rule-based pre-filter for trait signals."""

    def test_detects_new_interest(self):
        msg = "I started learning guitar last week, it's really fun"
        signals = detect_trait_signals(msg)
        assert len(signals) > 0
        assert 'started' in signals or 'learning' in signals

    def test_detects_life_change(self):
        msg = "I got promoted to senior engineer yesterday!"
        signals = detect_trait_signals(msg)
        assert 'promoted' in signals

    def test_detects_hobby(self):
        msg = "I've been really into woodworking lately"
        signals = detect_trait_signals(msg)
        assert 'really into' in signals

    def test_detects_quitting(self):
        msg = "I quit smoking three weeks ago"
        signals = detect_trait_signals(msg)
        assert 'quit' in signals

    def test_no_signals_in_casual_message(self):
        msg = "Hey, how are you doing today?"
        signals = detect_trait_signals(msg)
        assert len(signals) == 0

    def test_short_messages_filtered(self):
        msg = "hi"
        signals = detect_trait_signals(msg)
        assert len(signals) == 0

    def test_empty_message(self):
        signals = detect_trait_signals("")
        assert len(signals) == 0

    def test_none_message(self):
        signals = detect_trait_signals(None)
        assert len(signals) == 0


class TestMicroAdaptation:
    """Test the MicroAdaptation dataclass."""

    def test_to_dict(self):
        adaptation = MicroAdaptation(
            detected_trait="started learning guitar",
            trait_type="interest",
            user_or_companion="companion",
            suggested_adaptation="show curiosity about music",
            timestamp=NOW.isoformat(),
            source_message="I started learning guitar",
        )
        d = adaptation.to_dict()
        assert d['detected_trait'] == "started learning guitar"
        assert d['trait_type'] == "interest"
        assert 'timestamp' in d


class TestAdaptationStorage:
    """Test pending adaptation persistence."""

    def test_store_and_retrieve(self, tmp_path, monkeypatch):
        filepath = str(tmp_path / "adaptations.json")
        monkeypatch.setattr(
            'src.autonomy.persona_adaptation.ADAPTATIONS_FILE', filepath
        )

        adaptation = MicroAdaptation(
            detected_trait="got into cooking",
            trait_type="interest",
            user_or_companion="companion",
            suggested_adaptation="ask about recipes",
            timestamp=NOW.isoformat(),
            source_message="I got into cooking recently",
        )

        store_pending_adaptation(adaptation)
        pending = get_pending_adaptations()

        assert len(pending) == 1
        assert pending[0]['detected_trait'] == "got into cooking"

    def test_clear_adaptations(self, tmp_path, monkeypatch):
        filepath = str(tmp_path / "adaptations.json")
        monkeypatch.setattr(
            'src.autonomy.persona_adaptation.ADAPTATIONS_FILE', filepath
        )

        # Store one
        adaptation = MicroAdaptation(
            detected_trait="test",
            trait_type="interest",
            user_or_companion="companion",
            suggested_adaptation="test",
            timestamp=NOW.isoformat(),
            source_message="test",
        )
        store_pending_adaptation(adaptation)
        assert len(get_pending_adaptations()) == 1

        # Clear
        clear_pending_adaptations()
        assert len(get_pending_adaptations()) == 0

    def test_cap_at_max(self, tmp_path, monkeypatch):
        filepath = str(tmp_path / "adaptations.json")
        monkeypatch.setattr(
            'src.autonomy.persona_adaptation.ADAPTATIONS_FILE', filepath
        )
        monkeypatch.setattr(
            'src.autonomy.persona_adaptation.MAX_PENDING_ADAPTATIONS', 5
        )

        for i in range(10):
            adaptation = MicroAdaptation(
                detected_trait=f"trait {i}",
                trait_type="interest",
                user_or_companion="companion",
                suggested_adaptation=f"adapt {i}",
                timestamp=NOW.isoformat(),
                source_message=f"message {i}",
            )
            store_pending_adaptation(adaptation)

        pending = get_pending_adaptations()
        assert len(pending) <= 5


class TestValueInferenceFormatting:
    """Test formatting for macro-level integration."""

    def test_format_with_adaptations(self, tmp_path, monkeypatch):
        filepath = str(tmp_path / "adaptations.json")
        monkeypatch.setattr(
            'src.autonomy.persona_adaptation.ADAPTATIONS_FILE', filepath
        )

        adaptation = MicroAdaptation(
            detected_trait="started running",
            trait_type="lifestyle",
            user_or_companion="companion",
            suggested_adaptation="ask about runs",
            timestamp=NOW.isoformat(),
            source_message="I started running",
        )
        store_pending_adaptation(adaptation)

        formatted = format_adaptations_for_value_inference()
        assert "RECENT PERSONA SIGNALS" in formatted
        assert "started running" in formatted
        assert "lifestyle" in formatted

    def test_format_empty(self, tmp_path, monkeypatch):
        filepath = str(tmp_path / "adaptations_empty.json")
        monkeypatch.setattr(
            'src.autonomy.persona_adaptation.ADAPTATIONS_FILE', filepath
        )

        formatted = format_adaptations_for_value_inference()
        assert formatted == ""


class TestIndicatorCoverage:
    """Sanity checks on the indicator list."""

    def test_has_positive_indicators(self):
        positive = [i for i in TRAIT_INDICATORS if i in ['started', 'learning', 'got into', 'discovered']]
        assert len(positive) >= 3

    def test_has_negative_indicators(self):
        negative = [i for i in TRAIT_INDICATORS if i in ['quit', 'stopped', 'not anymore']]
        assert len(negative) >= 2

    def test_has_career_indicators(self):
        career = [i for i in TRAIT_INDICATORS if i in ['promoted', 'hired', 'fired', 'graduated']]
        assert len(career) >= 3
