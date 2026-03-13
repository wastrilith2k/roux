"""
Timezone Utilities -- Pacific Time helpers used throughout the framework.

WHAT: Provides `now_pacific()` (timezone-aware) and `now_pacific_naive()`
      (timezone-stripped) helpers plus a shared PACIFIC_TZ constant.

WHY:  The companion and user are both in Pacific Time. Every module that
      touches time should use these helpers instead of bare `datetime.now()`
      to avoid timezone bugs.

HOW:  Uses `zoneinfo.ZoneInfo("America/Los_Angeles")` which handles PST/PDT
      transitions automatically.
"""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# Pacific timezone (handles PST/PDT automatically)
PACIFIC_TZ = ZoneInfo("America/Los_Angeles")

def now_pacific():
    """Get current time in Pacific timezone."""
    return datetime.now(PACIFIC_TZ)

def now_pacific_naive():
    """Get current time in Pacific timezone as naive datetime (no tzinfo)."""
    return datetime.now(PACIFIC_TZ).replace(tzinfo=None)
