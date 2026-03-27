"""
Tests for thread-safe singleton initialization (Issue #52).

Verifies that the singleton factory functions in the four affected context
modules use double-checked locking so that concurrent threads always receive
the same instance.

Covers:
- get_presence_mode_manager()
- get_time_awareness()
- get_schedule_context_provider()
- get_derived_scene_context_builder()
"""

import os
import threading

os.environ.setdefault('ENVIRONMENT', 'test')
os.environ.setdefault('POSTGRES_PASSWORD', 'test')
os.environ.setdefault('POSTGRES_DB', 'test')
os.environ.setdefault('POSTGRES_HOST', 'localhost')
os.environ.setdefault('POSTGRES_PORT', '5432')
os.environ.setdefault('POSTGRES_USER', 'test')

import src.core.presence_mode as presence_mode_mod
import src.core.time_awareness as time_awareness_mod
import src.core.schedule_context as schedule_context_mod
import src.core.derived_scene_context as derived_scene_context_mod

from src.core.presence_mode import get_presence_mode_manager
from src.core.time_awareness import get_time_awareness
from src.core.schedule_context import get_schedule_context_provider
from src.core.derived_scene_context import get_derived_scene_context_builder


# Number of threads to race against each other.
NUM_THREADS = 20


def _race_factory(factory_fn, module, singleton_attr):
    """
    Call *factory_fn* from many threads simultaneously and return
    the set of distinct object ids produced.

    Resets the module-level singleton to None before each run so the
    race window is open.
    """
    setattr(module, singleton_attr, None)

    barrier = threading.Barrier(NUM_THREADS)
    results = [None] * NUM_THREADS

    def worker(idx):
        barrier.wait()  # all threads start together
        results[idx] = id(factory_fn())

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(NUM_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return set(results)


class TestPresenceModeSingleton:
    """get_presence_mode_manager() must return the same instance across threads."""

    def test_concurrent_access_returns_single_instance(self):
        ids = _race_factory(
            get_presence_mode_manager,
            presence_mode_mod,
            '_manager',
        )
        assert len(ids) == 1, f"Expected 1 instance, got {len(ids)}"


class TestTimeAwarenessSingleton:
    """get_time_awareness() must return the same instance across threads."""

    def test_concurrent_access_returns_single_instance(self):
        ids = _race_factory(
            get_time_awareness,
            time_awareness_mod,
            '_time_awareness',
        )
        assert len(ids) == 1, f"Expected 1 instance, got {len(ids)}"


class TestScheduleContextSingleton:
    """get_schedule_context_provider() must return the same instance across threads."""

    def test_concurrent_access_returns_single_instance(self):
        ids = _race_factory(
            get_schedule_context_provider,
            schedule_context_mod,
            '_provider',
        )
        assert len(ids) == 1, f"Expected 1 instance, got {len(ids)}"


class TestDerivedSceneContextSingleton:
    """get_derived_scene_context_builder() must return the same instance across threads."""

    def test_concurrent_access_returns_single_instance(self):
        ids = _race_factory(
            get_derived_scene_context_builder,
            derived_scene_context_mod,
            '_builder',
        )
        assert len(ids) == 1, f"Expected 1 instance, got {len(ids)}"
