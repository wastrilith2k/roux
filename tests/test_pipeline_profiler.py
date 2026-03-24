"""
Tests for pipeline profiling instrumentation (Issue #10, Task 2).

Covers:
- PipelineProfiler collects stage timings
- Sub-timings attach correctly
- Summary report is human-readable
- Edge cases (inactive profiler, non-existent stage)
- Per-request instantiation (not singleton)
"""

import os
import time
import pytest

os.environ.setdefault('ENVIRONMENT', 'test')
os.environ.setdefault('POSTGRES_PASSWORD', 'test')
os.environ.setdefault('POSTGRES_DB', 'test')
os.environ.setdefault('POSTGRES_HOST', 'localhost')
os.environ.setdefault('POSTGRES_PORT', '5432')
os.environ.setdefault('POSTGRES_USER', 'test')

from src.core.conversation.pipeline_profiler import PipelineProfiler, get_pipeline_profiler


class TestPipelineProfiler:
    """Test the profiler collects timing data correctly."""

    def test_basic_stage_timing(self):
        profiler = PipelineProfiler()
        profiler.start("How are you feeling?")

        with profiler.stage("context_assembly"):
            time.sleep(0.01)

        with profiler.stage("llm_call"):
            time.sleep(0.02)

        profile = profiler.finish()

        assert profile is not None
        assert len(profile.stages) == 2
        assert profile.stages[0].stage == "context_assembly"
        assert profile.stages[1].stage == "llm_call"
        assert profile.stages[0].duration_ms > 0
        assert profile.stages[1].duration_ms > 0
        assert profile.total_ms > 0

    def test_sub_timings_attached(self):
        profiler = PipelineProfiler()
        profiler.start("test message")

        with profiler.stage("context_assembly"):
            time.sleep(0.01)

        profiler.record_sub_timings("context_assembly", {
            "memories": 0.15,
            "entity_profiles": 0.08,
            "graphiti_context": 0.22,
            "personality": 0.03,
        })

        profile = profiler.finish()
        ctx_stage = profile.get_stage("context_assembly")
        assert ctx_stage is not None
        assert "memories" in ctx_stage.sub_timings
        assert ctx_stage.sub_timings["memories"] == pytest.approx(150, abs=1)
        assert ctx_stage.sub_timings["graphiti_context"] == pytest.approx(220, abs=1)

    def test_sub_timings_nonexistent_stage_ignored(self):
        """record_sub_timings on a stage that doesn't exist should be a no-op."""
        profiler = PipelineProfiler()
        profiler.start("test")

        with profiler.stage("real_stage"):
            pass

        # This should not raise or modify any stage
        profiler.record_sub_timings("nonexistent_stage", {"foo": 0.1})

        profile = profiler.finish()
        real = profile.get_stage("real_stage")
        assert real.sub_timings == {}  # Not contaminated

    def test_metadata_on_stages(self):
        profiler = PipelineProfiler()
        profiler.start("test")

        with profiler.stage("llm_call", model="fireworks/llama-3.1-405b"):
            pass

        profile = profiler.finish()
        assert profile.stages[0].metadata["model"] == "fireworks/llama-3.1-405b"

    def test_summary_report_format(self):
        profiler = PipelineProfiler()
        profiler.start("How are you feeling?")

        with profiler.stage("context_assembly"):
            time.sleep(0.01)

        with profiler.stage("memory_validation"):
            time.sleep(0.005)

        with profiler.stage("llm_call"):
            time.sleep(0.02)

        profiler.record_sub_timings("context_assembly", {
            "memories": 0.005,
            "graphiti_context": 0.003,
        })

        profile = profiler.finish()
        summary = profile.summary()

        assert "Pipeline Profile" in summary
        assert "context_assembly" in summary
        assert "memory_validation" in summary
        assert "llm_call" in summary
        assert "Bottleneck:" in summary
        assert "ms" in summary

    def test_get_stage_by_name(self):
        profiler = PipelineProfiler()
        profiler.start("test")

        with profiler.stage("foo"):
            pass
        with profiler.stage("bar"):
            pass

        profile = profiler.finish()
        assert profile.get_stage("foo") is not None
        assert profile.get_stage("bar") is not None
        assert profile.get_stage("baz") is None

    def test_inactive_profiler_yields_cleanly(self):
        profiler = PipelineProfiler()

        with profiler.stage("should_not_record"):
            pass

        profile = profiler.finish()
        assert profile is None

    def test_finish_returns_none_when_inactive(self):
        profiler = PipelineProfiler()
        assert profiler.finish() is None

    def test_bottleneck_identified(self):
        profiler = PipelineProfiler()
        profiler.start("test")

        with profiler.stage("fast"):
            time.sleep(0.001)
        with profiler.stage("slow"):
            time.sleep(0.03)

        profile = profiler.finish()
        summary = profile.summary()
        assert "Bottleneck: slow" in summary


class TestProfilerFactory:
    """Test that get_pipeline_profiler returns fresh instances (thread-safe)."""

    def test_returns_new_instance_each_call(self):
        p1 = get_pipeline_profiler()
        p2 = get_pipeline_profiler()
        assert p1 is not p2
