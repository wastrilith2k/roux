"""
Pipeline Profiler — Instruments the full message pipeline with per-stage timing.

WHAT: Collects wall-clock timing for every stage of message processing:
      STT transcription, memory retrieval (context building), context assembly,
      LLM call, tool execution, TTS generation, and delivery.
WHY:  Issue #10 Task 2 — measure where time is actually spent before optimizing.
      This profiler is read-only: it reports data but does not change behavior.
HOW:  PipelineProfiler wraps the existing pipeline stages with timing context
      managers. Results are logged and optionally returned as structured data.
      The profiler also pulls sub-timings from ContextBuilder._source_timings
      to break down the memory retrieval / context assembly stage.

Feature flag: COMPANION_PIPELINE_PROFILING_ENABLED (default: false)
"""

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

PROFILING_ENABLED = os.environ.get(
    'COMPANION_PIPELINE_PROFILING_ENABLED', 'false'
).lower() == 'true'


@dataclass
class StageTimingEntry:
    """Timing for a single pipeline stage."""
    stage: str
    duration_ms: float
    sub_timings: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineProfile:
    """Complete timing profile for one message through the pipeline."""
    message_preview: str
    total_ms: float
    stages: List[StageTimingEntry] = field(default_factory=list)

    def get_stage(self, name: str) -> Optional[StageTimingEntry]:
        for s in self.stages:
            if s.stage == name:
                return s
        return None

    def summary(self) -> str:
        """Human-readable timing summary."""
        lines = [
            f"Pipeline Profile ({self.total_ms:.0f}ms total) "
            f"for: \"{self.message_preview}\"",
            "-" * 60,
        ]

        for entry in self.stages:
            pct = (entry.duration_ms / self.total_ms * 100) if self.total_ms > 0 else 0
            bar = "#" * int(pct / 2)
            lines.append(f"  {entry.stage:<25} {entry.duration_ms:>7.0f}ms  ({pct:4.1f}%)  {bar}")

            # Show sub-timings if present (e.g., individual context sources)
            if entry.sub_timings:
                sorted_subs = sorted(entry.sub_timings.items(), key=lambda x: -x[1])
                for sub_name, sub_ms in sorted_subs[:5]:  # Top 5 slowest
                    lines.append(f"    └─ {sub_name:<21} {sub_ms:>7.0f}ms")

            # Show metadata
            for k, v in entry.metadata.items():
                lines.append(f"    ({k}: {v})")

        lines.append("-" * 60)

        # Identify bottleneck
        if self.stages:
            slowest = max(self.stages, key=lambda s: s.duration_ms)
            lines.append(f"  Bottleneck: {slowest.stage} ({slowest.duration_ms:.0f}ms)")

        return "\n".join(lines)


class PipelineProfiler:
    """Collects per-stage timing data during pipeline execution."""

    def __init__(self):
        self._stages: List[StageTimingEntry] = []
        self._start_time: float = 0.0
        self._active: bool = False

    def start(self, message_preview: str):
        """Begin profiling a new message."""
        self._stages = []
        self._start_time = time.time()
        self._message_preview = message_preview[:60]
        self._active = True

    @contextmanager
    def stage(self, name: str, **metadata):
        """Context manager to time a pipeline stage.

        Usage:
            with profiler.stage("llm_call", model="claude-3"):
                response = call_llm(...)
        """
        if not self._active:
            yield
            return

        start = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start) * 1000
            entry = StageTimingEntry(
                stage=name,
                duration_ms=duration_ms,
                metadata={k: str(v) for k, v in metadata.items()},
            )
            self._stages.append(entry)

    def record_sub_timings(self, stage_name: str, sub_timings: Dict[str, float]):
        """Attach sub-timings (e.g., context source timings) to a stage."""
        for entry in self._stages:
            if entry.stage == stage_name:
                entry.sub_timings = {
                    k: v * 1000 for k, v in sub_timings.items()  # Convert s -> ms
                }
                return

    def finish(self) -> Optional[PipelineProfile]:
        """Finish profiling and return the profile."""
        if not self._active:
            return None

        self._active = False
        total_ms = (time.time() - self._start_time) * 1000

        profile = PipelineProfile(
            message_preview=self._message_preview,
            total_ms=total_ms,
            stages=list(self._stages),
        )

        logger.info(f"\n{profile.summary()}")

        return profile


# Singleton
_profiler: Optional[PipelineProfiler] = None


def get_pipeline_profiler() -> PipelineProfiler:
    global _profiler
    if _profiler is None:
        _profiler = PipelineProfiler()
    return _profiler
