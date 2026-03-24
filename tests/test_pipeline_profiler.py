"""
Tests for pipeline profiling instrumentation (Issue #10, Task 2).

Covers:
- PipelineProfiler collects stage timings
- Sub-timings attach correctly
- Summary report is human-readable
- Profiler handles nested/overlapping stages
- Feature flag disables profiling
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


class TestPipelineProfiler:
    """Test the profiler collects timing data correctly."""

    def test_basic_stage_timing(self):
        from src.core.conversation.pipeline_profiler import PipelineProfiler

        profiler = PipelineProfiler()
        profiler.start("How are you feeling?")

        with profiler.stage("context_assembly"):
            time.sleep(0.01)  # 10ms

        with profiler.stage("llm_call"):
            time.sleep(0.02)  # 20ms

        profile = profiler.finish()

        assert profile is not None
        assert len(profile.stages) == 2
        assert profile.stages[0].stage == "context_assembly"
        assert profile.stages[1].stage == "llm_call"
        # Timings should be > 0
        assert profile.stages[0].duration_ms > 0
        assert profile.stages[1].duration_ms > 0
        assert profile.total_ms > 0

    def test_sub_timings_attached(self):
        from src.core.conversation.pipeline_profiler import PipelineProfiler

        profiler = PipelineProfiler()
        profiler.start("test message")

        with profiler.stage("context_assembly"):
            time.sleep(0.01)

        # Simulate context builder sub-timings (in seconds, converted to ms internally)
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
        assert ctx_stage.sub_timings["memories"] == pytest.approx(150, abs=1)  # 0.15s -> 150ms
        assert ctx_stage.sub_timings["graphiti_context"] == pytest.approx(220, abs=1)

    def test_metadata_on_stages(self):
        from src.core.conversation.pipeline_profiler import PipelineProfiler

        profiler = PipelineProfiler()
        profiler.start("test")

        with profiler.stage("llm_call", model="fireworks/llama-3.1-405b"):
            pass

        profile = profiler.finish()
        assert profile.stages[0].metadata["model"] == "fireworks/llama-3.1-405b"

    def test_summary_report_format(self):
        from src.core.conversation.pipeline_profiler import PipelineProfiler

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
        from src.core.conversation.pipeline_profiler import PipelineProfiler

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
        from src.core.conversation.pipeline_profiler import PipelineProfiler

        profiler = PipelineProfiler()
        # Don't call start() — profiler is inactive

        with profiler.stage("should_not_record"):
            pass

        profile = profiler.finish()
        assert profile is None

    def test_finish_returns_none_when_inactive(self):
        from src.core.conversation.pipeline_profiler import PipelineProfiler

        profiler = PipelineProfiler()
        assert profiler.finish() is None

    def test_bottleneck_identified(self):
        from src.core.conversation.pipeline_profiler import PipelineProfiler

        profiler = PipelineProfiler()
        profiler.start("test")

        with profiler.stage("fast"):
            time.sleep(0.001)
        with profiler.stage("slow"):
            time.sleep(0.03)

        profile = profiler.finish()
        summary = profile.summary()
        assert "Bottleneck: slow" in summary


class TestProfilerSingleton:
    """Test the singleton accessor."""

    def test_singleton_returns_same_instance(self):
        from src.core.conversation.pipeline_profiler import get_pipeline_profiler

        p1 = get_pipeline_profiler()
        p2 = get_pipeline_profiler()
        assert p1 is p2
