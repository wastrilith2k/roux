# Simulation Engine

[Back to Architecture Index](../ARCHITECTURE.md)

---

**Source**: `scripts/simulate_relationship.py`

The simulation system compresses weeks of relationship development into minutes. Two or more companions text each other, developing their relationship over time.

## How It Works

```
SimulationRunner
  -> SimulationClock (controls time globally)
  -> Daily loop:
       1. Energy reset based on time of day
       2. Life events fire randomly (configurable probability, default 0.6)
       3. 2-3 conversations selected as pairs
       4. Each conversation: 3-8 exchanges with unique conversation_id
       5. New openers use cross-conversation summaries
       6. End-of-day tasks (episodes, opinions, etc.)
  -> SimulationEventEmitter -> Redis pub/sub -> Observation Dashboard
  -> Weekly PostgreSQL checkpoint for review/rollback
```

## Simulation Configuration

**`instances/simulation_config.yaml`**:
- Models: configurable (default: Nemotron 3 Super 120B via OpenRouter free tier)
- 15 natural conversation opener seeds
- 12 tone modifiers (empty/neutral, teasing, contrarian, vulnerable, silly, protective, nostalgic, low-energy)
- 20 realistic daily activities
- 13 emotional mood states
- LLM settings: max_tokens=2000, temperature=0.95, summary temp=0.3

**`instances/simulation_config_fantasy.yaml`**:
- Fantasy-themed openers, activities, moods
- Setting reference: `settings/fantasy_tavern.yaml`
- Communication via "Sending Stones" (magical texting)

## Running Simulations

```bash
# Modern setting
python scripts/simulate_relationship.py --week 1

# Fantasy D&D party
python scripts/simulate_relationship.py --week 1 \
  --companions grimble lyra thorne kael \
  --config instances/simulation_config_fantasy.yaml

# Resume from checkpoint
python scripts/simulate_relationship.py --week 2

# Rollback to checkpoint
python scripts/simulate_relationship.py --rollback week1

# View summary
python scripts/simulate_relationship.py --summary
```

## Observation Dashboard

Real-time web dashboard at `/observe` for monitoring simulations:

**SocketIO events**: `sim:message`, `sim:day_start`, `sim:day_end`, `sim:conversation_start`, `sim:conversation_end`, `sim:task_complete`

**Features**:
- Real-time conversation feed with character color-coding
- Multi-select character filtering
- Collapsible state panels per character
- Tabs: Facts, Opinions, Curiosity, Goals, Episodes, Relationship metrics
