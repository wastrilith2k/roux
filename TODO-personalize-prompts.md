# TODO: Personalize hardcoded pronouns and names in prompts

## Context
The companion framework has hardcoded feminine pronouns ("she", "her") and
the user name "James" throughout LLM-facing prompts and internal comments.
This prevents the framework from being persona-agnostic. A companion could
be any gender, and the user won't always be "James."

## Step 1: Add pronoun fields to persona config

**File:** `data/persona.yaml` + `src/config/persona_config.py`

Add to `companion:` section in persona.yaml:
```yaml
companion:
  pronouns:
    subject: "she"      # she/he/they
    object: "her"       # her/him/them
    possessive: "her"   # her/his/their
    reflexive: "herself" # herself/himself/themselves
```

Add to `primary_user:` section:
```yaml
primary_user:
  pronouns:
    subject: "he"
    object: "him"
    possessive: "his"
    reflexive: "himself"
```

Add corresponding fields to `PersonaConfig` dataclass and wire them
in `_load_config()`. Provide sensible defaults (they/them).

## Step 2: Update LLM-facing prompts (high priority)

These strings are sent to the LLM and directly shape behavior. Fix first.

| File | Line(s) | Issue |
|------|---------|-------|
| `src/tasks/reflection_task.py` | 265 | `"writing her private reflections"` |
| `src/tasks/daily_summary_task.py` | 316 | `"writing her private daily journal"` |
| `src/tasks/episode_learning_task.py` | 163 | `"match his emotional needs"` |
| `src/autonomy/value_inference.py` | 223 | `"who she REALLY is"`, `"her values, fears"` |
| `src/autonomy/value_inference.py` | 417 | `"her values/boundaries"` |
| `src/autonomy/goal_formation.py` | 124, 265 | `"She should have"`, `"her own interests"` |
| `src/autonomy/opinion_store.py` | 395 | `"his emotional state"` |
| `src/autonomy/outcome_tracker.py` | 131 | `"He seemed disinterested"` |
| `src/autonomy/interjection_engine.py` | 465 | `"she was the last speaker"`, `"her own message"` |
| `src/autonomy/interjection_engine.py` | 525-531 | `"James said he was going to"` |
| `src/autonomy/reach_out_engine.py` | 834, 938 | `"her schedule"`, `"her own initiative"` |
| `src/autonomy/relationship_evaluation.py` | 2, 11, 20 | `"her own relationship status"`, `"her private determination"`, `"her feelings"` |
| `src/memory/core_memory.py` | 282 | `"writing her personal memory journal"` |
| `src/core/internal_state.py` | 1184-1186 | `"He said he was going to"` |
| `src/core/time_passage_narrator.py` | 244 | `"He was at his house"` |

## Step 3: Update internal comments/docstrings (low priority)

These don't affect behavior but should be updated for consistency.

| File | Lines | Issue |
|------|-------|-------|
| `src/tasks/reflection_task.py` | 15-16 | `"she notices"`, `"her own"` |
| `src/core/internal_state.py` | 534 | `"her internal clock"` |
| `src/core/background_life.py` | 5, 45, 144, 153, 161 | `"her schedule"`, `"she was doing"` |
| `src/core/time_passage_narrator.py` | 7 | `"her own thing"` |
| `src/core/proactive_curiosity.py` | 390 | `"she's been wondering"` |
| `src/autonomy/goals.py` | 5 | `"her own initiative"` |
| `src/autonomy/value_inference.py` | 4, 16 | `"her values"` |
| `src/autonomy/always_on_service.py` | 14 | `"her initiative"` |
| `src/memory/core_memory.py` | 6 | `"her private cheat sheet"` |

## Step 4: Create helper for pronoun interpolation

Rather than passing 4 pronoun fields everywhere, add a small helper:

```python
# src/config/persona_config.py

def companion_pronoun(self, form: str) -> str:
    """Get companion pronoun: 'subject', 'object', 'possessive', 'reflexive'"""
    return getattr(self, f'companion_pronoun_{form}')

def user_pronoun(self, form: str) -> str:
    """Get user pronoun: 'subject', 'object', 'possessive', 'reflexive'"""
    return getattr(self, f'user_pronoun_{form}')
```

Or a simple dict on PersonaConfig that templates can use:
```python
persona.companion_pronouns  # {"subject": "she", "object": "her", ...}
persona.user_pronouns       # {"subject": "he", "object": "him", ...}
```

## Done (already fixed)
- [x] `src/tasks/reflection_task.py` — `_generate_reflection` prompt: companion name + user name
- [x] `src/tasks/reflection_task.py` — `format_reflections_for_prompt`: companion name + user name
