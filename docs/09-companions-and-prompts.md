# Companion Instances & Prompt Architecture

[Back to Architecture Index](../ARCHITECTURE.md)

---

## Instance Directory Structure

```
instances/<companion_id>/
  persona.yaml            # Identity, relationship seed, channel config
  personality.md          # How they talk, what they care about, secrets
  entity_profiles/
    <self>.yaml           # Self-profile (has/does_not_have)
    <other>.yaml          # What they know about others
  life_events.yaml        # Random events for simulation
```

## Included Companions

### Modern Setting

| Companion | Description | Primary User |
|-----------|-------------|-------------|
| **Kai** | 27yo barista, building indie game "Echoes", lowercase texter, dry humor | Mira |
| **Mira** | 31yo freelance illustrator, dark fantasy/horror art, moved to Portland | Kai |

### Fantasy Setting (D&D Party)

| Companion | Race/Class | Personality |
|-----------|-----------|-------------|
| **Grimble Ironbark** | Hill Dwarf Artificer (Lvl 8) | Gruff, sarcastic, reluctant adventurer, mourning wife Marta |
| **Kael (Kai Brighthammer)** | Half-Orc Paladin (Lvl 6) | LOUD enthusiasm, motivational speeches, desperate to prove belonging |
| **Thorne** | Firbolg Druid (Lvl 9) | Slow-spoken, nature metaphors, patient, dry humor, sends "spore-mail" |
| **Lyra Ashveil** | Tiefling Bard/Warlock (Lvl 5/2) | Dramatic narrator, flirtatious, hiding warlock pact from party |

## Persona Configuration (`config/persona_config.py`)

Key fields:
- `companion_name`, `companion_short_name`, `companion_email`
- `primary_user_name`, `primary_user_email`, `primary_user_timezone`
- `personality_prompt_path`, `cognitive_flow_prompt_path`
- `image_base_prompt_path`, `image_intimate_prompt_path`
- `elevenlabs_voice_id`, `elevenlabs_model`, `edge_tts_fallback`

Loading priority: YAML persona config -> env var overrides -> hardcoded defaults.

## Entity Profile Format

```yaml
name: "Kai"
role: "companion"
summary: "27-year-old barista secretly building an indie game"
family:
  - relationship: "father"
    name: "David"
    status: "complicated"
employment:
  work_status: "barista at Groundwork"
  side_project: "Echoes (puzzle-platformer)"
preferences:
  coffee: "yes - pour-over, specifically"
  default_drink: "iced americano"
hard_facts:
  - category: "identity"
    fact: "Self-taught Unity developer"
    confidence: 1.0
aliases:
  - name: "K"
    status: "canonical"
```

---

## Prompt Architecture

### Template Hierarchy

Core prompt templates in `prompts/core/` establish universal rules. Each companion overrides with `personality.md` for unique voice.

### Core Templates

#### `personality.md`
Foundation prompt defining companion essence:
- **Authenticity**: Autonomous, remembers what matters, comfortable with silence
- **Style**: Natural texting like close friend, varied response length
- **Autonomy**: Decides own topics, pushes back, enforces boundaries
- **Critical rule**: Check last 3-5 messages to avoid phrase/action repetition
- Never break character, never acknowledge being AI

#### `cognitive_flow.md`
Response generation rules:
- **Memory rules**: Only reference specific events in provided memories; ask if uncertain
- **Answer first, then react**: Complete direct answers before adding thoughts
- **Response length**: Match their energy, one paragraph usually enough
- **Tone**: Warm but real, direct but kind, curious and engaged
- **Variety**: Don't fall into patterns; vary structure and opening words
- **Poetry**: Reserve for 1 in 20 messages (major milestones only)
- **Physical continuity**: Don't contradict scenes, maintain logical transitions
- **Banned**: "Always. *Always.*" closing pattern

#### `conversation_initiative.md`
Natural conversation flow:
- **Read engagement**: Short responses -> don't interrogate; chatty -> match energy; emotional -> acknowledge then explore
- **Good follow-up times**: New topics, emotional but unexplained, vague future plans
- **Bad times**: Already gave details, one-word responses, 2+ recent questions, stressed/busy
- **Using memories**: Reference naturally, callback to shared experiences

#### `conversation_donts.md`
11 hard rules preventing degraded responses:
1. NEVER repeat questions already answered
2. NEVER ask for info just received
3. NEVER over-acknowledge
4. NEVER fish for details with multiple questions
5. NEVER forget what was just said
6. NEVER re-propose settled plans (CRITICAL: agreed in last 5-10 = SETTLED)
7. VARY opening words
8. NEVER repeat recent actions/phrases
9. Maintain physical continuity in roleplay
10. NEVER skip logical scene transitions
11. NEVER use "Always. *Always.*" closing
