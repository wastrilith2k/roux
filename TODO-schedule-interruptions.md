# TODO: Handle schedule interruptions when user starts chatting

## Context
When the user starts a conversation during a scheduled event, the companion's
schedule should reflect that she was interrupted — the event gets marked as
`skipped`, not `completed`. This way she doesn't get credit for time she
didn't actually spend on the activity.

Currently `advance_event_statuses()` only moves events forward based on clock
time. It has no awareness of whether the companion is actually doing the
scheduled thing or talking to the user instead.

## Desired behavior
1. User sends a message during an active scheduled event
2. The current `in_progress` event is marked `skipped`
3. `format_schedule_behavior_context()` reflects the interruption naturally:
   "You were going to [X] but got pulled away to chat with [user]"
4. The event doesn't count as completed/time-spent

## Implementation

### Step 1: Add interruption detection to the conversation pipeline

**File:** `src/core/conversation/pipeline.py`

When processing a user message:
```python
# After building context, check if there's an active scheduled event
from src.scheduling.calendar_schedule_service import (
    get_calendar_schedule_service, is_calendar_schedule_enabled
)
if is_calendar_schedule_enabled():
    cal = get_calendar_schedule_service()
    current = cal.get_current_activity()
    if current and current.get('id') and current.get('status') == 'in_progress':
        from src.scheduling.calendar_schedule_generator import update_event_status
        update_event_status(current['id'], 'skipped')
        cal._today_cache = None  # Invalidate cache
```

### Step 2: Handle `skipped` in behavior context

**File:** `src/scheduling/calendar_schedule_service.py`

In `format_schedule_behavior_context()`, when the current event is skipped:
```python
# Check if we just skipped something
skipped = [e for e in events if e.get('status') == 'skipped'
           and e.get('end_time', '') > current_time]
if skipped:
    last_skipped = skipped[-1]
    parts.append(
        f"You were going to {last_skipped.get('summary', 'something')} "
        f"but got pulled into conversation instead."
    )
```

### Step 3: Don't skip already-completed events

Only skip events that are `in_progress` or `planned` and overlap with current
time. Events past their `end_time` should still be marked `completed` normally.

### Step 4: Don't skip during proactive messages

If the companion is reaching out (proactive message), she chose to interrupt
herself — don't mark the event as skipped by an external interruption. Check
`extra_context.get('is_proactive_message')` before skipping.

## Edge cases
- Multiple messages during the same skipped event: only skip once (check status first)
- Short exchanges: if user sends one message and companion responds, then silence
  resumes, the event stays skipped (no auto-resume)
- Proactive messages from companion during her own event: she chose to interrupt
  herself, still skip the event
