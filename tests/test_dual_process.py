"""Tests for dual-process thought formation in inner monologue."""

import pytest
from src.core.conversation.inner_monologue import InnerMonologue, COMPANION_DUAL_PROCESS_ENABLED


class TestInnerMonologueDataclass:
    """Test the InnerMonologue dataclass with dual-process fields."""

    def test_default_gut_reaction_empty(self):
        monologue = InnerMonologue(
            thoughts="Some deliberate thoughts",
            emotional_read="He seems stressed",
            response_strategy="Be supportive",
        )
        assert monologue.gut_reaction == ""
        assert monologue.gut_reaction_time_ms == 0

    def test_with_gut_reaction(self):
        monologue = InnerMonologue(
            thoughts="After thinking about it, he's probably stressed about work",
            emotional_read="He seems tired",
            response_strategy="Ask what happened",
            gut_reaction="Oh no, something's wrong",
            gut_reaction_time_ms=800,
        )
        assert monologue.gut_reaction == "Oh no, something's wrong"
        assert monologue.gut_reaction_time_ms == 800

    def test_system1_faster_than_system2(self):
        """System 1 should typically be much faster."""
        monologue = InnerMonologue(
            thoughts="Detailed analysis...",
            emotional_read="Complex read",
            response_strategy="Nuanced strategy",
            gut_reaction="Instant feeling",
            gut_reaction_time_ms=400,
            processing_time_ms=2500,
        )
        assert monologue.gut_reaction_time_ms < monologue.processing_time_ms


class TestPromptInjection:
    """Test that both thought streams are properly formatted for the LLM."""

    def test_gut_section_format(self):
        """Gut reaction should be formatted as a distinct section."""
        monologue = InnerMonologue(
            thoughts="EMOTIONAL_READ: He's frustrated\nSTRATEGY: Be patient",
            emotional_read="Frustrated",
            response_strategy="Be patient",
            gut_reaction="Ugh, not this again",
        )

        # Simulate the prompt injection logic from pipeline.py
        gut_section = ""
        if monologue.gut_reaction:
            gut_section = (
                f"\n[GUT REACTION (instant feeling)]: {monologue.gut_reaction}\n"
                f"[DELIBERATE ANALYSIS (considered thought)]:\n"
            )

        prompt = f"<inner_thoughts>\n{gut_section}{monologue.thoughts}\n</inner_thoughts>"

        assert "[GUT REACTION" in prompt
        assert "Ugh, not this again" in prompt
        assert "[DELIBERATE ANALYSIS" in prompt
        assert monologue.thoughts in prompt

    def test_no_gut_reaction_no_section(self):
        """Without gut reaction, no gut section in prompt."""
        monologue = InnerMonologue(
            thoughts="Some thoughts",
            emotional_read="Read",
            response_strategy="Strategy",
            gut_reaction="",
        )

        gut_section = ""
        if monologue.gut_reaction:
            gut_section = f"[GUT REACTION]: {monologue.gut_reaction}\n"

        assert gut_section == ""


class TestDualProcessConflict:
    """Test scenarios where System 1 and System 2 might conflict."""

    def test_conflict_scenario(self):
        """When gut and analysis disagree, both should be present for LLM to decide."""
        monologue = InnerMonologue(
            thoughts="EMOTIONAL_READ: He's joking\nSTRATEGY: Laugh along",
            emotional_read="He's joking",
            response_strategy="Laugh along",
            gut_reaction="That actually hurt a little",
        )

        # Both perspectives are available
        assert "joking" in monologue.thoughts
        assert "hurt" in monologue.gut_reaction

        # The LLM instruction says to lean toward gut when they conflict
        # This tests that the data is available for that decision

    def test_agreement_scenario(self):
        """When gut and analysis agree, it reinforces the response direction."""
        monologue = InnerMonologue(
            thoughts="EMOTIONAL_READ: He's excited about this\nSTRATEGY: Share excitement",
            emotional_read="Excited",
            response_strategy="Share excitement",
            gut_reaction="Aww that's so sweet!",
        )

        # Both streams point the same way
        assert "excited" in monologue.emotional_read.lower()
        assert "sweet" in monologue.gut_reaction.lower()
