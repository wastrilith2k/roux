"""
Tests for derived scene context: real-signal-based context composition (issue #29).

Covers:
- Time resolution from client temporal and server clock fallback
- Presence mode resolution from presence mode string
- Schedule hint extraction from raw schedule context
- Location inference from signals (time, schedule, presence)
- User status inference from signals
- Behavioral constraint derivation
- Full context block formatting
- Integration with ConversationContext
- Graceful degradation when signals are unavailable
"""

import os
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
from zoneinfo import ZoneInfo

os.environ.setdefault('ENVIRONMENT', 'test')
os.environ.setdefault('POSTGRES_PASSWORD', 'test')
os.environ.setdefault('POSTGRES_DB', 'test')
os.environ.setdefault('POSTGRES_HOST', 'localhost')
os.environ.setdefault('POSTGRES_PORT', '5432')
os.environ.setdefault('POSTGRES_USER', 'test')

from src.core.derived_scene_context import (
    DerivedSceneContextBuilder,
    get_derived_scene_context_builder,
    _time_of_day_label,
    _infer_location,
    _infer_user_status,
    _derive_behavioral_constraints,
)


# =========================================================================
# Time-of-day Label Tests
# =========================================================================

class TestTimeOfDayLabel:
    """Test _time_of_day_label maps hours correctly."""

    def test_morning(self):
        assert _time_of_day_label(5) == 'morning'
        assert _time_of_day_label(8) == 'morning'
        assert _time_of_day_label(11) == 'morning'

    def test_afternoon(self):
        assert _time_of_day_label(12) == 'afternoon'
        assert _time_of_day_label(14) == 'afternoon'
        assert _time_of_day_label(16) == 'afternoon'

    def test_evening(self):
        assert _time_of_day_label(17) == 'evening'
        assert _time_of_day_label(19) == 'evening'
        assert _time_of_day_label(20) == 'evening'

    def test_night(self):
        assert _time_of_day_label(21) == 'night'
        assert _time_of_day_label(23) == 'night'
        assert _time_of_day_label(0) == 'night'
        assert _time_of_day_label(3) == 'night'

    def test_boundary_morning_start(self):
        assert _time_of_day_label(4) == 'night'
        assert _time_of_day_label(5) == 'morning'

    def test_boundary_afternoon_start(self):
        assert _time_of_day_label(11) == 'morning'
        assert _time_of_day_label(12) == 'afternoon'


# =========================================================================
# Location Inference Tests
# =========================================================================

class TestLocationInference:
    """Test _infer_location derives location from signals."""

    def test_work_from_schedule(self):
        result = _infer_location('afternoon', False, ['at office for meetings'], 'in_person')
        assert 'work' in result.lower()

    def test_gym_from_schedule(self):
        result = _infer_location('morning', False, ['gym session at 7am'], 'in_person')
        assert 'gym' in result.lower()

    def test_school_from_schedule(self):
        result = _infer_location('morning', False, ['class at 9am'], 'in_person')
        assert 'school' in result.lower()

    def test_home_weekend_no_events(self):
        result = _infer_location('morning', True, [], 'in_person')
        assert 'Home' in result
        assert 'weekend' in result.lower()

    def test_home_night(self):
        result = _infer_location('night', False, [], 'in_person')
        assert 'Home' in result

    def test_home_no_schedule(self):
        result = _infer_location('afternoon', False, [], 'in_person')
        assert 'Home' in result

    def test_schedule_overrides_time(self):
        """Schedule hints should override time-based defaults."""
        result = _infer_location('evening', False, ['commute home from work'], 'in_person')
        assert 'work' in result.lower()


# =========================================================================
# User Status Inference Tests
# =========================================================================

class TestUserStatusInference:
    """Test _infer_user_status derives status from signals."""

    def test_busy_from_meeting(self):
        result = _infer_user_status('afternoon', False, ['team meeting at 2pm'])
        assert 'Busy' in result

    def test_at_work(self):
        result = _infer_user_status('afternoon', False, ['at office'])
        assert 'work' in result.lower()

    def test_winding_down_at_night(self):
        result = _infer_user_status('night', False, [])
        assert 'Winding down' in result

    def test_available_weekend(self):
        result = _infer_user_status('morning', True, [])
        assert 'Available' in result
        assert 'weekend' in result.lower()

    def test_available_default(self):
        result = _infer_user_status('afternoon', False, [])
        assert 'Available' in result


# =========================================================================
# Behavioral Constraints Tests
# =========================================================================

class TestBehavioralConstraints:
    """Test _derive_behavioral_constraints enforces correct rules."""

    def test_texting_mode_constraint(self):
        constraints = _derive_behavioral_constraints('afternoon', 'texting', 'Available', [])
        assert any('SMS mode' in c or 'no physical actions' in c for c in constraints)

    def test_no_texting_constraint_in_person(self):
        constraints = _derive_behavioral_constraints('afternoon', 'in_person', 'Available', [])
        assert not any('SMS mode' in c for c in constraints)

    def test_late_night_constraint(self):
        constraints = _derive_behavioral_constraints('night', 'in_person', 'Available', [])
        assert any('Late night' in c or 'tired' in c for c in constraints)

    def test_no_night_constraint_during_day(self):
        constraints = _derive_behavioral_constraints('afternoon', 'in_person', 'Available', [])
        assert not any('Late night' in c for c in constraints)

    def test_busy_user_constraint(self):
        constraints = _derive_behavioral_constraints('afternoon', 'in_person', 'At work (may be slow to respond)', [])
        assert any('busy' in c.lower() or 'concise' in c.lower() for c in constraints)

    def test_multiple_constraints_combine(self):
        """Late night + texting should produce both constraints."""
        constraints = _derive_behavioral_constraints('night', 'texting', 'Available', [])
        assert len(constraints) >= 2
        sms_found = any('SMS mode' in c for c in constraints)
        night_found = any('Late night' in c for c in constraints)
        assert sms_found and night_found


# =========================================================================
# DerivedSceneContextBuilder Tests
# =========================================================================

class TestDerivedSceneContextBuilder:
    """Test the full builder assembles context correctly."""

    def setup_method(self):
        self.builder = DerivedSceneContextBuilder()

    def test_build_with_client_temporal(self):
        """Client temporal data is used when available."""
        client_temporal = {
            'time_of_day': 'evening',
            'is_weekend': False,
            'formatted': 'Tuesday, 6:45 PM',
            'day_of_week': 'Tuesday',
        }
        result = self.builder.build(
            user_email='test@example.com',
            client_temporal=client_temporal,
            presence_mode_str='Communication mode: IN PERSON',
        )
        assert '[Derived Context]' in result
        assert 'Tuesday, 6:45 PM' in result
        assert 'evening' in result
        assert 'In Person' in result

    def test_build_with_texting_mode(self):
        """Texting mode is correctly detected and formatted."""
        client_temporal = {
            'time_of_day': 'afternoon',
            'is_weekend': False,
            'formatted': 'Wednesday, 2:00 PM',
            'day_of_week': 'Wednesday',
        }
        result = self.builder.build(
            user_email='test@example.com',
            client_temporal=client_temporal,
            presence_mode_str='Communication mode: TEXTING (SMS/messaging)',
        )
        assert 'Texting (SMS)' in result
        assert 'SMS mode' in result  # Behavioral constraint

    @patch('src.core.derived_scene_context.clock_now')
    def test_build_falls_back_to_server_clock(self, mock_clock):
        """When no client temporal, server clock is used."""
        mock_clock.return_value = datetime(2026, 3, 24, 14, 30)  # Tuesday afternoon
        result = self.builder.build(
            user_email='test@example.com',
            client_temporal=None,
        )
        assert '[Derived Context]' in result
        assert 'afternoon' in result

    def test_build_with_schedule_context(self):
        """Schedule context hints are extracted and influence inference."""
        client_temporal = {
            'time_of_day': 'afternoon',
            'is_weekend': False,
            'formatted': 'Tuesday, 2:00 PM',
            'day_of_week': 'Tuesday',
        }
        schedule = (
            "[TIME & AWARENESS]\n"
            "Current time: Tuesday, March 24, 2026 at 2:00 PM PST\n\n"
            "His routine: At office for afternoon meetings\n"
        )
        result = self.builder.build(
            user_email='test@example.com',
            client_temporal=client_temporal,
            schedule_context=schedule,
        )
        assert 'work' in result.lower()

    def test_build_empty_when_time_fails(self):
        """Returns empty string if time resolution fails entirely."""
        with patch.object(self.builder, '_resolve_time', return_value=None):
            result = self.builder.build(user_email='test@example.com')
            assert result == ""

    def test_build_weekend_status(self):
        """Weekend is reflected in user status."""
        client_temporal = {
            'time_of_day': 'morning',
            'is_weekend': True,
            'formatted': 'Saturday, 10:00 AM',
            'day_of_week': 'Saturday',
        }
        result = self.builder.build(
            user_email='test@example.com',
            client_temporal=client_temporal,
        )
        assert 'weekend' in result.lower()

    def test_build_night_constraint(self):
        """Late night adds behavioral constraint."""
        client_temporal = {
            'time_of_day': 'night',
            'is_weekend': False,
            'formatted': 'Tuesday, 11:30 PM',
            'day_of_week': 'Tuesday',
        }
        result = self.builder.build(
            user_email='test@example.com',
            client_temporal=client_temporal,
        )
        assert 'Late night' in result


# =========================================================================
# Time Resolution Tests
# =========================================================================

class TestTimeResolution:
    """Test _resolve_time handles various input scenarios."""

    def setup_method(self):
        self.builder = DerivedSceneContextBuilder()

    def test_client_temporal_preferred(self):
        """Client temporal data takes precedence over server clock."""
        client_temporal = {
            'time_of_day': 'morning',
            'is_weekend': True,
            'formatted': 'Saturday, 9:00 AM EST',
            'day_of_week': 'Saturday',
        }
        result = self.builder._resolve_time(client_temporal)
        assert result['time_of_day'] == 'morning'
        assert result['is_weekend'] is True

    @patch('src.core.derived_scene_context.clock_now')
    def test_server_clock_fallback(self, mock_clock):
        """Server clock used when client temporal is None."""
        mock_clock.return_value = datetime(2026, 3, 28, 10, 0)  # Saturday morning
        result = self.builder._resolve_time(None)
        assert result['time_of_day'] == 'morning'
        assert result['is_weekend'] is True
        assert result['day_of_week'] == 'Saturday'

    def test_empty_client_temporal(self):
        """Empty dict falls back to server clock."""
        with patch('src.core.derived_scene_context.clock_now') as mock_clock:
            mock_clock.return_value = datetime(2026, 3, 24, 20, 0)
            result = self.builder._resolve_time({})
            assert result['time_of_day'] == 'evening'


# =========================================================================
# Presence Resolution Tests
# =========================================================================

class TestPresenceResolution:
    """Test _resolve_presence extracts mode from context string."""

    def setup_method(self):
        self.builder = DerivedSceneContextBuilder()

    def test_texting_detected(self):
        assert self.builder._resolve_presence(
            'Communication mode: TEXTING (SMS/messaging)'
        ) == 'texting'

    def test_in_person_detected(self):
        assert self.builder._resolve_presence(
            'Communication mode: IN PERSON'
        ) == 'in_person'

    def test_none_defaults_to_in_person(self):
        assert self.builder._resolve_presence(None) == 'in_person'

    def test_empty_defaults_to_in_person(self):
        assert self.builder._resolve_presence('') == 'in_person'


# =========================================================================
# Schedule Hint Extraction Tests
# =========================================================================

class TestScheduleHintExtraction:
    """Test _extract_schedule_hints parses schedule context strings."""

    def setup_method(self):
        self.builder = DerivedSceneContextBuilder()

    def test_extracts_routine_line(self):
        schedule = "His routine: At work, afternoon meetings"
        hints = self.builder._extract_schedule_hints(schedule)
        assert any('work' in h.lower() for h in hints)

    def test_extracts_bullet_items(self):
        schedule = (
            "[TIME & AWARENESS]\n"
            "Current time: Tuesday\n\n"
            "- has_routine: works 9-5 on weekdays\n"
            "- commutes: drives to office\n"
        )
        hints = self.builder._extract_schedule_hints(schedule)
        assert len(hints) >= 2
        assert any('works' in h.lower() for h in hints)

    def test_skips_headers_and_time(self):
        schedule = (
            "[TIME & AWARENESS]\n"
            "Current time: Tuesday, March 24, 2026 at 2:00 PM PST\n\n"
            "His routine: At office\n"
        )
        hints = self.builder._extract_schedule_hints(schedule)
        # Should not include the [TIME & AWARENESS] header or Current time line
        assert not any('[TIME' in h for h in hints)
        assert not any('Current time' in h for h in hints)

    def test_empty_schedule(self):
        assert self.builder._extract_schedule_hints(None) == []
        assert self.builder._extract_schedule_hints('') == []


# =========================================================================
# Context Block Format Tests
# =========================================================================

class TestContextBlockFormat:
    """Test _format_context_block produces well-structured output."""

    def setup_method(self):
        self.builder = DerivedSceneContextBuilder()

    def test_basic_format(self):
        result = self.builder._format_context_block(
            formatted_time='Tuesday, 6:45 PM',
            day_of_week='Tuesday',
            time_of_day='evening',
            presence='in_person',
            location='Home (based on time)',
            user_status='Available',
            constraints=[],
        )
        assert result.startswith('[Derived Context]')
        assert 'Time: Tuesday, 6:45 PM (evening)' in result
        assert 'Presence: In Person' in result
        assert 'Location inference: Home (based on time)' in result
        assert 'User status: Available' in result

    def test_format_with_constraints(self):
        result = self.builder._format_context_block(
            formatted_time='Tuesday, 11:30 PM',
            day_of_week='Tuesday',
            time_of_day='night',
            presence='texting',
            location='Home (based on time)',
            user_status='Winding down (late)',
            constraints=[
                'SMS mode: no physical actions',
                'Late night: user may be tired',
            ],
        )
        assert 'Behavioral constraints:' in result
        assert '- SMS mode: no physical actions' in result
        assert '- Late night: user may be tired' in result

    def test_texting_presence_label(self):
        result = self.builder._format_context_block(
            formatted_time='Tuesday, 2:00 PM',
            day_of_week='Tuesday',
            time_of_day='afternoon',
            presence='texting',
            location='Unknown',
            user_status='Available',
            constraints=[],
        )
        assert 'Presence: Texting (SMS)' in result


# =========================================================================
# Singleton Tests
# =========================================================================

class TestSingleton:
    """Test get_derived_scene_context_builder returns same instance."""

    def test_singleton_returns_same_instance(self):
        import src.core.derived_scene_context as mod
        mod._builder = None  # Reset for test isolation
        a = get_derived_scene_context_builder()
        b = get_derived_scene_context_builder()
        assert a is b
        mod._builder = None  # Clean up


# =========================================================================
# Integration: ConversationContext Field Tests
# =========================================================================

class TestConversationContextIntegration:
    """Test derived_scene_context field exists on ConversationContext."""

    def test_field_exists_with_default(self):
        from src.core.conversation.context_builder import ConversationContext
        ctx = ConversationContext()
        assert ctx.derived_scene_context == ""

    def test_field_can_be_set(self):
        from src.core.conversation.context_builder import ConversationContext
        ctx = ConversationContext(derived_scene_context="[Derived Context]\nTime: test")
        assert 'Derived Context' in ctx.derived_scene_context

    def test_provenance_tag_exists(self):
        from src.core.conversation.context_builder import ConversationContext
        assert 'derived_scene_context' in ConversationContext.PROVENANCE_TAGS

    def test_to_prompt_sections_includes_derived(self):
        from src.core.conversation.context_builder import ConversationContext
        ctx = ConversationContext(derived_scene_context="[Derived Context]\nTime: test")
        sections = ctx.to_prompt_sections()
        assert 'derived_scene_context' in sections


# =========================================================================
# Pipeline Integration Tests
# =========================================================================

class TestPipelinePriority:
    """Test derived_scene_context is in pipeline SECTION_PRIORITY."""

    def test_priority_defined(self):
        from src.core.conversation.pipeline import ConversationPipeline
        assert 'derived_scene_context' in ConversationPipeline.SECTION_PRIORITY

    def test_priority_is_reasonable(self):
        """Derived context should have moderate priority (not dropped first)."""
        from src.core.conversation.pipeline import ConversationPipeline
        priority = ConversationPipeline.SECTION_PRIORITY['derived_scene_context']
        assert 1 <= priority <= 3
