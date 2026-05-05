#!/usr/bin/env python3
"""
simulate_adventure.py — D&D Mini-Adventure: The Missing Kin

A party-adventure format simulator where a Narrator drives the scene and all 4
characters (Grimble, Kael, Lyra, Thorne) respond in-person. Unlike
simulate_relationship.py (text-message format), this script generates an
immersive, prose-driven fantasy adventure.

Usage:
    python scripts/simulate_adventure.py
    python scripts/simulate_adventure.py --dry-run
    python scripts/simulate_adventure.py --skip-wipe
    python scripts/simulate_adventure.py --start-scene 4
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------

def find_project_root() -> Path:
    candidates = [Path(__file__).parent.parent, Path('/app'), Path.cwd()]
    for p in candidates:
        if (p / 'data').exists() or (p / 'instances').exists():
            return p
    raise RuntimeError("Cannot find project root")


PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_log_handlers = [logging.StreamHandler()]
try:
    logs_dir = PROJECT_ROOT / 'logs'
    logs_dir.mkdir(exist_ok=True)
    _log_handlers.append(logging.FileHandler(logs_dir / 'adventure.log', mode='a'))
except OSError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=_log_handlers,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Character personalities (loaded once at module level)
# ---------------------------------------------------------------------------

CHARACTERS = ['grimble', 'kael', 'lyra', 'thorne']

CHARACTER_EMAILS = {
    'grimble': 'grimble@companion.local',
    'kael':    'kael@companion.local',
    'lyra':    'lyra@companion.local',
    'thorne':  'thorne@companion.local',
}

CHARACTER_DISPLAY_NAMES = {
    'grimble': 'Grimble',
    'kael':    'Kael',
    'lyra':    'Lyra',
    'thorne':  'Thorne',
}

# SRD stat blocks
STAT_BLOCKS = {
    'grimble': {'ac': 16, 'hp': 72, 'atk_bonus': 6,  'dmg_dice': (1, 8),  'dmg_bonus': 4,  'dex_mod': 0},
    'kael':    {'ac': 20, 'hp': 62, 'atk_bonus': 8,  'dmg_dice': (1, 8),  'dmg_bonus': 6,  'dex_mod': 0,
                'smite_dice': (2, 8), 'smite_available': True},
    'lyra':    {'ac': 13, 'hp': 30, 'atk_bonus': 7,  'dmg_dice': (1, 10), 'dmg_bonus': 5,  'dex_mod': 2,
                'ranged': True},
    'thorne':  {'ac': 14, 'hp': 66, 'atk_bonus': 9,  'dmg_dice': (1, 6),  'dmg_bonus': 0,  'dex_mod': -1,
                'ranged': True,
                'bear_ac': 11, 'bear_hp': 34, 'bear_atk': 5, 'bear_dmg_dice': (2, 6), 'bear_dmg_bonus': 4},
    'goblin':  {'ac': 15, 'hp': 7,  'atk_bonus': 4,  'dmg_dice': (1, 6),  'dmg_bonus': 2,  'dex_mod': 2},
    'snagtooth': {'ac': 16, 'hp': 21, 'atk_bonus': 5, 'dmg_dice': (2, 6), 'dmg_bonus': 3,  'dex_mod': 1,
                  'attacks_per_round': 2},
}


def load_personality(character_id: str) -> str:
    path = PROJECT_ROOT / 'instances' / character_id / 'personality.md'
    if path.exists():
        return path.read_text()
    logger.warning(f"No personality.md found for {character_id}")
    return f"You are {CHARACTER_DISPLAY_NAMES.get(character_id, character_id)}."


PERSONALITIES = {cid: load_personality(cid) for cid in CHARACTERS}

# ---------------------------------------------------------------------------
# Dice and combat mechanics
# ---------------------------------------------------------------------------

def dice(n: int, sides: int) -> int:
    """Roll n d-sides and return the sum."""
    return sum(random.randint(1, sides) for _ in range(n))


def resolve_attack(atk_bonus: int, target_ac: int) -> tuple[bool, int]:
    """Roll d20 + atk_bonus vs target_ac. Returns (hit, roll_total)."""
    roll = random.randint(1, 20)
    total = roll + atk_bonus
    return total >= target_ac, total


def build_combat_summary(rounds: list[dict]) -> str:
    """Build the structured combat summary string to feed to the Narrator.

    Each round dict has:
      round_num, actions: list of {actor, target, roll, total, ac, hit, dmg, dmg_rolled, note}
      result: str describing outcome
    """
    lines = []
    for r in rounds:
        lines.append(f"COMBAT ROUND {r['round_num']}:")
        for a in r['actions']:
            if a.get('skip'):
                lines.append(f"  - {a['actor']}: {a.get('note', 'no action')}")
                continue
            hit_str = "HIT" if a['hit'] else "MISS"
            dmg_str = f", {a['dmg_rolled']}={a['dmg']} damage" if a['hit'] else ""
            note_str = f" ({a['note']})" if a.get('note') else ""
            lines.append(
                f"  - {a['actor']} attacks {a['target']}: "
                f"roll {a['roll']}+{a['atk_bonus']}={a['total']} vs AC{a['ac']} "
                f"→ {hit_str}{dmg_str}{note_str}"
            )
        lines.append(f"  RESULT: {r['result']}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_db_conn():
    """Return a direct psycopg2 connection using env vars."""
    import psycopg2
    return psycopg2.connect(
        host=os.environ.get('POSTGRES_HOST', 'postgres'),
        port=os.environ.get('POSTGRES_PORT', '5432'),
        dbname=os.environ.get('POSTGRES_DB', 'companion_dev'),
        user=os.environ.get('POSTGRES_USER', 'companion'),
        password=os.environ.get('POSTGRES_PASSWORD', 'companion_dev_password'),
    )


def wipe_old_data():
    """DELETE all adventure messages from all 4 party schemas."""
    import psycopg2
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            for cid in CHARACTERS:
                schema = f"user_{cid}"
                try:
                    cur.execute(
                        f"DELETE FROM {schema}.messages WHERE source = 'adventure'"
                    )
                    logger.info(f"Wiped adventure messages from {schema}.messages")
                except psycopg2.Error as e:
                    logger.warning(f"Could not wipe {schema}.messages: {e}")
                    conn.rollback()
        conn.commit()
    finally:
        conn.close()


def get_redis():
    """Return a Redis client, or None if unavailable."""
    try:
        import redis
        r = redis.Redis(
            host=os.environ.get('REDIS_HOST', 'redis'),
            port=6379,
            decode_responses=True,
        )
        r.ping()
        return r
    except Exception as e:
        logger.warning(f"Redis unavailable: {e}")
        return None


_redis_client = None


def _get_redis_cached():
    global _redis_client
    if _redis_client is None:
        _redis_client = get_redis()
    return _redis_client


def store_message(
    sender: str,
    content: str,
    conv_id: int,
    timestamp: datetime,
    dry_run: bool = False,
):
    """Insert message into all 4 party schemas and publish to Redis."""
    if dry_run:
        print(f"\n[{timestamp.strftime('%H:%M')}] {sender}: {content}\n")
        return

    import psycopg2
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            for cid in CHARACTERS:
                email = CHARACTER_EMAILS[cid]
                schema = f"user_{cid}"
                audience = list(CHARACTER_EMAILS.values())
                try:
                    cur.execute(
                        f"""INSERT INTO {schema}.messages
                            (email, sender_name, message_text, timestamp, source, conversation_id, audience)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                        (
                            email,
                            sender,
                            content,
                            timestamp,
                            'adventure',
                            conv_id,
                            audience,
                        ),
                    )
                except psycopg2.Error as e:
                    logger.warning(f"Failed to insert into {schema}.messages: {e}")
                    conn.rollback()
        conn.commit()
    finally:
        conn.close()

    # Redis publish
    r = _get_redis_cached()
    if r:
        try:
            r.publish(
                'simulation_events',
                json.dumps({
                    'type': 'sim:message',
                    'data': {
                        'speaker': sender,
                        'content': content,
                        '_ts': timestamp.isoformat(),
                    },
                }),
            )
        except Exception as e:
            logger.debug(f"Redis publish failed: {e}")

    # Print to stdout for live observation
    print(f"\n[{timestamp.strftime('%H:%M')}] {sender}: {content}\n")


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

NARRATOR_SYSTEM = (
    "You are the narrator for a gritty, grounded fantasy adventure. "
    "Your prose is vivid but economical — like a good novelist, not a video game announcer. "
    "You describe what characters see, smell, feel. You make danger feel real. "
    "You never use game or tabletop terminology (no HP, AC, rolls, damage, saving throws, etc.) "
    "— only natural prose. Present tense. "
    "When narrating combat, focus on the physicality and emotion of the fight."
)

CHARACTER_SYSTEM_TEMPLATE = (
    "You are {name}, responding in real-time during an adventure. "
    "You are physically present in this scene — not texting, not writing a letter. "
    "Speak and act as yourself. 2-3 sentences max: your words and/or actions. "
    "Stay completely in character. No meta-commentary.\n\n"
    "{personality}"
)


def _openrouter_client():
    """Return an openai.OpenAI client pointed at OpenRouter."""
    from openai import OpenAI
    return OpenAI(
        base_url='https://openrouter.ai/api/v1',
        api_key=os.environ.get('OPENROUTER_API_KEY', ''),
    )


def narrator_llm(prompt: str) -> str:
    """Call OpenRouter Kimi-K2 as the literary narrator. Returns narration text."""
    client = _openrouter_client()
    resp = client.chat.completions.create(
        model='moonshotai/kimi-k2',
        messages=[
            {'role': 'system', 'content': NARRATOR_SYSTEM},
            {'role': 'user',   'content': prompt},
        ],
        max_tokens=600,
        temperature=0.85,
    )
    return resp.choices[0].message.content.strip()


def character_llm(character_id: str, scene_context: str, action_prompt: str) -> str:
    """Generate a character's in-scene response. Returns dialogue/action text."""
    name = CHARACTER_DISPLAY_NAMES[character_id]
    personality = PERSONALITIES[character_id]
    system = CHARACTER_SYSTEM_TEMPLATE.format(name=name, personality=personality)

    client = _openrouter_client()
    resp = client.chat.completions.create(
        model='moonshotai/kimi-k2',
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user',   'content': f"{scene_context}\n\n{action_prompt}"},
        ],
        max_tokens=200,
        temperature=0.9,
    )
    text = resp.choices[0].message.content.strip()
    # Strip leaked name prefixes like "Grimble:" or "**Kael:**"
    import re
    text = re.sub(rf'^\*{{0,2}}{re.escape(name)}\*{{0,2}}\s*[:]\s*', '', text, count=1)
    return text.strip()


def sleep_between_calls(n: int = 4):
    """Sleep n seconds to avoid rate limits."""
    time.sleep(n)


# ---------------------------------------------------------------------------
# Dry-run stubs
# ---------------------------------------------------------------------------

DRY_RUN_NARRATIONS = {
    'narrator': "[Narrator narration placeholder — would be generated by LLM]",
    'character': "[Character response placeholder — would be generated by LLM]",
}


def narrator_dry(scene_label: str) -> str:
    return f"[NARRATOR — {scene_label}] {DRY_RUN_NARRATIONS['narrator']}"


def character_dry(character_id: str, scene_label: str) -> str:
    name = CHARACTER_DISPLAY_NAMES[character_id]
    return f"[{name} — {scene_label}] {DRY_RUN_NARRATIONS['character']}"


# ---------------------------------------------------------------------------
# Adventure runner
# ---------------------------------------------------------------------------

class AdventureRunner:
    """Runs all 9 scenes of The Missing Kin adventure."""

    WORLD_START = datetime(2026, 2, 1, 12, 0)  # noon, Feb 1 2026

    def __init__(self, dry_run: bool = False, skip_wipe: bool = False, start_scene: int = 1):
        self.dry_run   = dry_run
        self.skip_wipe = skip_wipe
        self.start_scene = start_scene
        self.clock     = self.WORLD_START
        # Track which characters have used their one smite
        self._kael_smite_used = False
        # Track if Thorne has bear-shifted this fight
        self._thorne_bear = False

    # ------------------------------------------------------------------
    # Clock helpers
    # ------------------------------------------------------------------

    def advance_clock(self, minutes: int = 0, hours: int = 0):
        self.clock += timedelta(minutes=minutes, hours=hours)

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def emit(self, sender: str, content: str, conv_id: int):
        store_message(sender, content, conv_id, self.clock, dry_run=self.dry_run)

    def narrator(self, prompt: str, conv_id: int, label: str = '') -> str:
        if self.dry_run:
            text = narrator_dry(label or 'scene')
        else:
            sleep_between_calls(4)
            text = narrator_llm(prompt)
        self.emit('Narrator', text, conv_id)
        return text

    def advance_turn(self, minutes: int = 2):
        """Advance the world clock between character speaking turns."""
        self.advance_clock(minutes=minutes)

    def character(self, character_id: str, scene_context: str, action_prompt: str,
                  conv_id: int, label: str = '',
                  prior: list = None) -> str:
        """Generate and store a character response.

        prior: list of (name, text) tuples from earlier in the same scene.
               Appended to scene_context so the character can react to what
               was just said rather than independently responding to the same prompt.
        """
        ctx = scene_context
        if prior:
            lines = '\n'.join(f"{name}: {text}" for name, text in prior[-3:])
            ctx = ctx + f"\n\nWhat has just been said:\n{lines}"

        if self.dry_run:
            text = character_dry(character_id, label or 'response')
        else:
            sleep_between_calls(4)
            text = character_llm(character_id, ctx, action_prompt)
        self.emit(CHARACTER_DISPLAY_NAMES[character_id], text, conv_id)
        return text

    # ------------------------------------------------------------------
    # Combat engine
    # ------------------------------------------------------------------

    def _run_combat(
        self,
        conv_id: int,
        heroes: list[str],
        enemies: list[dict],  # list of {id, name, hp, ac, atk_bonus, dmg_dice, dmg_bonus, dex_mod}
        min_rounds: int = 2,
        thorne_can_bear: bool = False,
        kael_can_smite: bool = False,
        combat_label: str = 'combat',
    ) -> list[dict]:
        """Run a full combat encounter. Returns list of round dicts for the Narrator."""
        # Reset per-combat bear state only (smite is once per fight, tracked globally)
        if thorne_can_bear:
            self._thorne_bear = False

        # Mutable hero HP pools
        hero_hp = {h: STAT_BLOCKS[h]['hp'] for h in heroes}
        enemy_pools = [{**e} for e in enemies]  # copy so we can mutate hp

        # Initiative order: (dex_mod + d20, name)
        def roll_initiative(name, dex_mod):
            return random.randint(1, 20) + dex_mod, name

        rounds = []
        round_num = 0

        while True:
            round_num += 1

            # Build initiative order for this round
            initiative = []
            for h in heroes:
                if hero_hp.get(h, 0) > 0:
                    init_roll, _ = roll_initiative(h, STAT_BLOCKS[h]['dex_mod'])
                    initiative.append((init_roll, h, 'hero'))
            for e in enemy_pools:
                if e['hp'] > 0:
                    init_roll, _ = roll_initiative(e['name'], e['dex_mod'])
                    initiative.append((init_roll, e['id'], 'enemy'))
            initiative.sort(key=lambda x: -x[0])

            actions = []
            remaining_enemies = [e for e in enemy_pools if e['hp'] > 0]
            remaining_heroes  = [h for h in heroes if hero_hp.get(h, 0) > 0]

            for init_val, actor_id, actor_type in initiative:
                if actor_type == 'hero':
                    # Choose a living enemy target
                    living_enemies = [e for e in enemy_pools if e['hp'] > 0]
                    if not living_enemies:
                        break

                    target_e = living_enemies[0]
                    stats = STAT_BLOCKS[actor_id]

                    # Thorne might drop into bear form
                    use_bear = False
                    if actor_id == 'thorne' and thorne_can_bear and not self._thorne_bear:
                        # Bear form: shift happens if thorne is still up and enemies are dangerous
                        if len(living_enemies) >= 2 or round_num >= 2:
                            self._thorne_bear = True
                            use_bear = True

                    if use_bear:
                        atk_bonus = stats['bear_atk']
                        dmg_dice  = stats['bear_dmg_dice']
                        dmg_bonus = stats['bear_dmg_bonus']
                        note = "Thorne shifts into bear form — claws"
                    else:
                        atk_bonus = stats['atk_bonus']
                        dmg_dice  = stats['dmg_dice']
                        dmg_bonus = stats['dmg_bonus']
                        note = "ranged" if stats.get('ranged') else ""

                    hit, total = resolve_attack(atk_bonus, target_e['ac'])
                    dmg = 0
                    dmg_rolled_str = ""

                    if hit:
                        base_dmg = dice(*dmg_dice) + dmg_bonus

                        # Kael's smite: once per whole adventure
                        smite_note = ""
                        if actor_id == 'kael' and kael_can_smite and not self._kael_smite_used:
                            smite_dmg = dice(*stats['smite_dice'])
                            base_dmg += smite_dmg
                            self._kael_smite_used = True
                            smite_note = f" + smite {stats['smite_dice'][0]}d{stats['smite_dice'][1]}={smite_dmg} radiant"
                            note = f"DIVINE SMITE{smite_note}"
                            dmg_rolled_str = f"{dmg_dice[0]}d{dmg_dice[1]}+{dmg_bonus}+smite={base_dmg}"
                        else:
                            dmg_rolled_str = f"{dmg_dice[0]}d{dmg_dice[1]}+{dmg_bonus}={base_dmg}"

                        dmg = base_dmg
                        prev_hp = target_e['hp']
                        target_e['hp'] -= dmg
                        if target_e['hp'] <= 0:
                            note_suffix = f" ({target_e['name']} was at {prev_hp} HP: DOWN)"
                            note = (note + note_suffix) if note else note_suffix.strip()

                    actions.append({
                        'actor':     CHARACTER_DISPLAY_NAMES.get(actor_id, actor_id),
                        'target':    target_e['name'],
                        'roll':      total - atk_bonus,
                        'atk_bonus': atk_bonus,
                        'total':     total,
                        'ac':        target_e['ac'],
                        'hit':       hit,
                        'dmg':       dmg,
                        'dmg_rolled': dmg_rolled_str,
                        'note':      note,
                    })

                else:
                    # Enemy attacks a random living hero
                    living_heroes = [h for h in heroes if hero_hp.get(h, 0) > 0]
                    if not living_heroes:
                        break

                    target_h = random.choice(living_heroes)
                    e_stats = next((e for e in enemy_pools if e['id'] == actor_id), None)
                    if not e_stats or e_stats['hp'] <= 0:
                        continue

                    hit, total = resolve_attack(e_stats['atk_bonus'], STAT_BLOCKS[target_h]['ac'])
                    dmg = 0
                    dmg_rolled_str = ""
                    note = ""
                    if hit:
                        dmg = dice(*e_stats['dmg_dice']) + e_stats['dmg_bonus']
                        dmg_rolled_str = f"{e_stats['dmg_dice'][0]}d{e_stats['dmg_dice'][1]}+{e_stats['dmg_bonus']}={dmg}"
                        hero_hp[target_h] = max(0, hero_hp[target_h] - dmg)
                        if hero_hp[target_h] == 0:
                            note = f"{CHARACTER_DISPLAY_NAMES[target_h]} is down!"

                    actions.append({
                        'actor':     e_stats['name'],
                        'target':    CHARACTER_DISPLAY_NAMES[target_h],
                        'roll':      total - e_stats['atk_bonus'],
                        'atk_bonus': e_stats['atk_bonus'],
                        'total':     total,
                        'ac':        STAT_BLOCKS[target_h]['ac'],
                        'hit':       hit,
                        'dmg':       dmg,
                        'dmg_rolled': dmg_rolled_str,
                        'note':      note,
                    })

            # Check end conditions
            living_enemies = [e for e in enemy_pools if e['hp'] > 0]
            living_heroes  = [h for h in heroes if hero_hp.get(h, 0) > 0]

            if not living_enemies:
                result = f"All enemies defeated. Fight ended in {round_num} round(s)."
                if use_bear if 'use_bear' in dir() else False:
                    result += " Thorne remains in bear form, growling."
                rounds.append({'round_num': round_num, 'actions': actions, 'result': result})
                break

            if not living_heroes:
                result = "The party is overwhelmed — things look dire."
                rounds.append({'round_num': round_num, 'actions': actions, 'result': result})
                break

            fallen = [e['name'] for e in enemy_pools if e['hp'] <= 0]
            standing = [e['name'] for e in living_enemies]
            result = f"Still fighting. Fallen: {', '.join(fallen) if fallen else 'none'}. Remaining: {', '.join(standing)}."
            rounds.append({'round_num': round_num, 'actions': actions, 'result': result})

            # Enforce minimum rounds even if enemies die early (re-check after min_rounds)
            if round_num >= min_rounds and not living_enemies:
                break
            if round_num >= 10:  # safety cap
                break

        return rounds

    # ------------------------------------------------------------------
    # Scene implementations
    # ------------------------------------------------------------------

    def run_scene_1(self):
        """Scene 1: Thornwall Market — midday resupply."""
        CONV_ID = 100
        logger.info("=== Scene 1: Thornwall Market ===")

        narration_prompt = (
            "Set the opening scene of our adventure: midday at Thornwall Market. "
            "It's a cold late-winter day, the 1st of February, just past noon. "
            "Thornwall's lower-quarter market is busy — merchants hawking their wares, "
            "the smell of roasting meat and horse dung and fresh-cut wood. "
            "The party (Grimble the dwarf artificer, Kael the half-orc paladin, "
            "Lyra the tiefling bard, and Thorne the firbolg druid) are each running "
            "their own errands before they meet back up. Set the scene vividly. 4-5 sentences."
        )
        self.narrator(narration_prompt, CONV_ID, label='market-open')

        scene_ctx = (
            "It is midday, 1 February. The party is resupplying at Thornwall Market — "
            "the lower quarter, busy and cold. Each of you is on a separate errand. "
            "The market smells of smoke, animal, and winter."
        )

        # Four separate errands — no chaining, they're apart from each other.
        # Advance 1 minute between each so they have distinct timestamps.
        self.character(
            'grimble', scene_ctx,
            "Grimble needs metal stock or tinkering components. Show him haggling badly "
            "with a merchant, or muttering at his own hammer. 2-3 sentences.",
            CONV_ID, label='market-grimble',
        )
        self.advance_turn(minutes=1)

        self.character(
            'lyra', scene_ctx,
            "Lyra is at a fabric or jewellery stall, or loitering near a street musician "
            "she could do better than. She overhears something or spots something she wants. "
            "2-3 sentences.",
            CONV_ID, label='market-lyra',
        )
        self.advance_turn(minutes=1)

        self.character(
            'kael', scene_ctx,
            "Kael is buying food for the group — too much of it. He stops to help a "
            "struggling vendor or greets a stranger warmly. Show his size and his enthusiasm. "
            "2-3 sentences.",
            CONV_ID, label='market-kael',
        )
        self.advance_turn(minutes=1)

        self.character(
            'thorne', scene_ctx,
            "Thorne is buying dried herbs or mushrooms and is deeply uncomfortable in the crowd. "
            "He finds a quiet moment at the edge of a stall — presses his palm to cobblestone, "
            "feels something. 2-3 sentences.",
            CONV_ID, label='market-thorne',
        )

    def run_scene_2(self):
        """Scene 2: The Plea — farmer Aldric approaches."""
        CONV_ID = 100
        logger.info("=== Scene 2: The Plea ===")

        narration_prompt = (
            "The party has finished their errands and is gathering near the market's "
            "central well. The market is quieting, late afternoon light going orange. "
            "An old farmer — weathered, 60s, desperate — approaches the group. "
            "His name is Aldric. He has clearly been searching for someone to help. "
            "He stops in front of the party, hat in both hands. 4-5 sentences."
        )
        self.narrator(narration_prompt, CONV_ID, label='plea-open')
        self.advance_turn(minutes=1)

        # Aldric speaks
        aldric_prompt = (
            "Generate Aldric's plea to the party. He is a weathered farmer in his 60s, "
            "desperate, barely holding himself together. His grandson Tomas, age 9, "
            "was taken from the Ashwood road two nights ago by goblins. "
            "He's been searching for help and couldn't afford proper mercenaries. "
            "He's proud and it costs him something to beg. "
            "He describes what he saw on the road. 3-4 sentences, first person."
        )
        if self.dry_run:
            aldric_text = narrator_dry('Aldric plea')
        else:
            sleep_between_calls(4)
            aldric_text = narrator_llm(aldric_prompt)
        self.emit('Aldric', aldric_text, CONV_ID)

        scene_ctx = (
            "You are gathered at the market well. An old farmer named Aldric has just told you: "
            "his grandson Tomas (age 9) was taken by goblins on the Ashwood road two nights ago. "
            "He's desperate and nearly out of hope. He can't pay much. Tomas is still alive — "
            "goblins take prisoners for trade or worse."
        )

        # Three characters react — chained so each builds on the last.
        prior = []

        self.advance_turn(minutes=1)
        text = self.character(
            'kael', scene_ctx,
            "Kael reacts first — immediately, genuinely moved. He probably speaks "
            "before anyone else can stop him. 2-3 sentences.",
            CONV_ID, label='plea-kael', prior=prior,
        )
        prior.append(('Kael', text))

        self.advance_turn(minutes=1)
        text = self.character(
            'grimble', scene_ctx,
            "Grimble reacts to what Kael just said and to the farmer's story. "
            "He's skeptical of the details, practical — but he's listening. 2-3 sentences.",
            CONV_ID, label='plea-grimble', prior=prior,
        )
        prior.append(('Grimble', text))

        self.advance_turn(minutes=1)
        text = self.character(
            'lyra', scene_ctx,
            "Lyra has been watching Aldric's face while the others talked. "
            "She asks a pointed question or makes an unexpected observation. 2-3 sentences.",
            CONV_ID, label='plea-lyra', prior=prior,
        )

        self.advance_clock(minutes=10)

    def run_scene_3(self):
        """Scene 3: The Trail — into Ashwood forest."""
        CONV_ID = 100
        logger.info("=== Scene 3: The Trail ===")

        narration_prompt = (
            "The party has left Thornwall and is following goblin tracks into the Ashwood — "
            "a dense forest that smells of wet pine and old rot. The tracks are fresh, "
            "pressed deep into mud. It's mid-afternoon but the tree cover makes it feel like dusk. "
            "Thorne is reading the forest floor. Lyra keeps watch behind. "
            "Something feels wrong — the birds have gone quiet. 4-5 sentences."
        )
        self.narrator(narration_prompt, CONV_ID, label='trail-open')

        scene_ctx = (
            "The party is moving through Ashwood forest, following goblin tracks. "
            "It's mid-afternoon but the tree cover is thick. The birds have gone quiet. "
            "Thorne is leading, reading the ground. Something feels wrong about this forest today."
        )

        prior = []

        self.advance_turn(minutes=2)
        text = self.character(
            'thorne', scene_ctx,
            "Thorne is reading the tracks and the forest itself. He senses something wrong "
            "beyond just goblins — the earth is sick, or afraid. He shares what he's reading. "
            "2-3 sentences.",
            CONV_ID, label='trail-thorne', prior=prior,
        )
        prior.append(('Thorne', text))

        self.advance_turn(minutes=2)
        text = self.character(
            'lyra', scene_ctx,
            "Lyra is keeping watch at the back of the party. She reacts to what Thorne said "
            "and to something she hears or feels. She's performing bravery she doesn't quite have. "
            "2-3 sentences.",
            CONV_ID, label='trail-lyra', prior=prior,
        )
        prior.append(('Lyra', text))

        self.advance_turn(minutes=2)
        self.character(
            'grimble', scene_ctx,
            "Grimble has been squinting at the tracks through his goggles. "
            "He adds something practical — about the spacing of the tracks, or what they suggest "
            "about what's ahead. 2-3 sentences.",
            CONV_ID, label='trail-grimble', prior=prior,
        )

        self.advance_clock(minutes=75)

    def run_scene_4(self):
        """Scene 4: The Cave — discovering the goblin den."""
        CONV_ID = 100
        logger.info("=== Scene 4: The Cave ===")

        narration_prompt = (
            "Describe the goblin den: a hillside with a crude timber door, "
            "smoke leaking from gaps in the wood, the smell of rot and wet fur and old cook-fire. "
            "Two goblin sentries are visible outside — one asleep against the door, "
            "one whittling something and glancing around. The tracks lead straight here. "
            "The party is watching from the treeline, thirty feet back. 4-5 sentences."
        )
        self.narrator(narration_prompt, CONV_ID, label='cave-open')

        scene_ctx = (
            "The party is watching from the treeline, thirty feet from the goblin den. "
            "Two sentries outside the crude timber door. Tomas is presumably inside. "
            "You are whispering about how to approach."
        )

        prior = []

        self.advance_turn(minutes=2)
        text = self.character(
            'kael', scene_ctx,
            "Kael wants to go straight in — he's thinking about the boy. "
            "He proposes something direct, maybe impulsive. 2-3 sentences.",
            CONV_ID, label='cave-kael', prior=prior,
        )
        prior.append(('Kael', text))

        self.advance_turn(minutes=2)
        text = self.character(
            'grimble', scene_ctx,
            "Grimble is studying the door and the hinges with a craftsman's eye. "
            "He responds to Kael's plan — probably with a critique — and proposes something "
            "more methodical. 2-3 sentences.",
            CONV_ID, label='cave-grimble', prior=prior,
        )
        prior.append(('Grimble', text))

        self.advance_turn(minutes=2)
        self.character(
            'thorne', scene_ctx,
            "Thorne has been staring at the door and the tight cave entrance behind it. "
            "His fear of enclosed spaces is surfacing. He says something practical but there's "
            "something underneath it. He's already dreading going in. 2-3 sentences.",
            CONV_ID, label='cave-thorne', prior=prior,
        )

        self.advance_clock(minutes=20)

    def run_scene_5(self):
        """Scene 5 (conv 104): First Blood — fight the 2 entrance sentries."""
        CONV_ID = 100
        logger.info("=== Scene 5: First Blood ===")

        heroes = ['kael', 'grimble', 'lyra', 'thorne']
        enemies = [
            {'id': 'goblin_a', 'name': 'Goblin Sentry A', 'hp': 7,  'ac': 15,
             'atk_bonus': 4, 'dmg_dice': (1, 6), 'dmg_bonus': 2, 'dex_mod': 2},
            {'id': 'goblin_b', 'name': 'Goblin Sentry B', 'hp': 7,  'ac': 15,
             'atk_bonus': 4, 'dmg_dice': (1, 6), 'dmg_bonus': 2, 'dex_mod': 2},
        ]

        rounds = self._run_combat(
            conv_id=CONV_ID,
            heroes=heroes,
            enemies=enemies,
            min_rounds=1,
            thorne_can_bear=False,
            kael_can_smite=False,
            combat_label='sentries',
        )

        combat_summary = build_combat_summary(rounds)
        total_rounds = len(rounds)

        narration_prompt = (
            f"The party has sprung their ambush on the two goblin sentries. "
            f"Here is what happened mechanically — translate this into pure, vivid prose. "
            f"NO game terms whatsoever. Describe it as physical action, sound, sensation. "
            f"Make the goblins feel genuinely dangerous even if the fight was quick. "
            f"The fight took {total_rounds} round(s). 4-5 sentences of combat narration.\n\n"
            f"MECHANICS (for your reference only — do NOT include these in your output):\n"
            f"{combat_summary}"
        )
        self.narrator(narration_prompt, CONV_ID, label='sentries-combat')

        self.advance_turn(minutes=2)
        # One character reacts post-fight
        scene_ctx = (
            "The two entrance sentries are down. The party stands at the cave door. "
            "Inside, there are sounds of movement — the fight may have been heard. "
            f"The combat took {total_rounds} round(s)."
        )
        self.character(
            'kael', scene_ctx,
            "Kael is catching his breath after the fight. A quick reaction — something "
            "about the boy, or a word of readiness. 2-3 sentences.",
            CONV_ID, label='sentries-kael',
        )

    def run_scene_6(self):
        """Scene 6 (conv 105): Through the Den — 4 goblins in 2 sub-groups."""
        CONV_ID = 100
        logger.info("=== Scene 6: Through the Den ===")

        heroes = ['kael', 'grimble', 'lyra', 'thorne']

        # First sub-group: 2 goblins
        enemies_1 = [
            {'id': 'goblin_c', 'name': 'Goblin Guard C', 'hp': 7, 'ac': 15,
             'atk_bonus': 4, 'dmg_dice': (1, 6), 'dmg_bonus': 2, 'dex_mod': 2},
            {'id': 'goblin_d', 'name': 'Goblin Guard D', 'hp': 7, 'ac': 15,
             'atk_bonus': 4, 'dmg_dice': (1, 6), 'dmg_bonus': 2, 'dex_mod': 2},
        ]

        # Second sub-group: 2 more goblins
        enemies_2 = [
            {'id': 'goblin_e', 'name': 'Goblin Scout E', 'hp': 7, 'ac': 15,
             'atk_bonus': 4, 'dmg_dice': (1, 6), 'dmg_bonus': 2, 'dex_mod': 2},
            {'id': 'goblin_f', 'name': 'Goblin Scout F', 'hp': 7, 'ac': 15,
             'atk_bonus': 4, 'dmg_dice': (1, 6), 'dmg_bonus': 2, 'dex_mod': 2},
        ]

        # Narrate entering the den
        entry_prompt = (
            "The party pushes through the crude timber door into the goblin den. "
            "Describe the interior: low ceiling, torches guttering in sconces of lashed bone, "
            "the air thick with smoke and animal smell. The passages are narrow — barely wide enough "
            "for Thorne's shoulders. Two goblins spot them immediately. 4-5 sentences. "
            "Thorne's claustrophobia should be palpable in the description."
        )
        self.narrator(entry_prompt, CONV_ID, label='den-entry')

        # First fight
        rounds_1 = self._run_combat(
            conv_id=CONV_ID,
            heroes=heroes,
            enemies=enemies_1,
            min_rounds=1,
            combat_label='den-fight-1',
        )
        summary_1 = build_combat_summary(rounds_1)

        combat_narration_1 = (
            f"Narrate the first fight inside the den in vivid prose. NO game terms. "
            f"Physical action only. The tight corridor changes how the fight feels — "
            f"Kael can't swing freely, Thorne has almost no room. Lyra's magic lights up the passage. "
            f"Grimble is efficient and brutal. 4-5 sentences.\n\n"
            f"MECHANICS (reference only):\n{summary_1}"
        )
        self.narrator(combat_narration_1, CONV_ID, label='den-fight-1-narration')

        self.advance_turn(minutes=2)
        # Character reaction after first fight — chained pair
        scene_ctx_mid = (
            "The first two goblins inside are down. The passage ahead splits. "
            "The party is catching their breath in a torch-lit corridor. "
            "The ceiling is low. Thorne's shoulders almost touch the walls."
        )
        prior_mid = []
        text = self.character(
            'thorne', scene_ctx_mid,
            "Thorne's claustrophobia is pressing down on him in this tunnel. "
            "He says something — maybe to himself, maybe to the party. "
            "He keeps moving but it costs him something. 2-3 sentences.",
            CONV_ID, label='den-thorne', prior=prior_mid,
        )
        prior_mid.append(('Thorne', text))

        self.advance_turn(minutes=1)
        # Thorne discovers signs of Tomas
        discovery_prompt = (
            "Thorne notices something: disturbed bedding of straw and rags in a side alcove, "
            "and a small torn piece of grey cloth — a child's tunic, recently. "
            "Narrate the discovery. Thorne recognizes the scent of a living human child, recent. "
            "The boy was definitely here. 3-4 sentences."
        )
        self.narrator(discovery_prompt, CONV_ID, label='den-discovery')

        self.advance_turn(minutes=1)
        self.character(
            'grimble', scene_ctx_mid,
            "Grimble sees the torn cloth and reacts to what Thorne and the narrator have revealed. "
            "Gruff — but something shows through. He picks up the cloth or checks for traps. "
            "2-3 sentences.",
            CONV_ID, label='den-grimble-cloth', prior=prior_mid,
        )

        # Second fight
        rounds_2 = self._run_combat(
            conv_id=CONV_ID,
            heroes=heroes,
            enemies=enemies_2,
            min_rounds=2,
            combat_label='den-fight-2',
        )
        summary_2 = build_combat_summary(rounds_2)

        combat_narration_2 = (
            f"Narrate the second fight in the den — this group of goblins was ready. "
            f"One of them screams for reinforcements before it goes down. "
            f"The party is getting tired. Someone takes a hit. NO game terms. 4-5 sentences.\n\n"
            f"MECHANICS (reference only):\n{summary_2}"
        )
        self.narrator(combat_narration_2, CONV_ID, label='den-fight-2-narration')

        self.advance_clock(minutes=30)

    def run_scene_7(self):
        """Scene 7 (conv 106): Snagtooth's Chamber — boss fight."""
        CONV_ID = 100
        logger.info("=== Scene 7: Snagtooth's Chamber ===")

        heroes = ['kael', 'grimble', 'lyra', 'thorne']

        # Boss + 2 bodyguard goblins
        enemies = [
            {'id': 'snagtooth', 'name': 'Warlord Snagtooth', 'hp': 21, 'ac': 16,
             'atk_bonus': 5, 'dmg_dice': (2, 6), 'dmg_bonus': 3, 'dex_mod': 1,
             'attacks_per_round': 2},
            {'id': 'bodyguard_1', 'name': 'Goblin Bodyguard 1', 'hp': 7, 'ac': 15,
             'atk_bonus': 4, 'dmg_dice': (1, 6), 'dmg_bonus': 2, 'dex_mod': 2},
            {'id': 'bodyguard_2', 'name': 'Goblin Bodyguard 2', 'hp': 7, 'ac': 15,
             'atk_bonus': 4, 'dmg_dice': (1, 6), 'dmg_bonus': 2, 'dex_mod': 2},
        ]

        # Entering the chamber
        entry_prompt = (
            "The passage opens into a wider chamber — the ceiling vaults upward. "
            "In the center stands Warlord Snagtooth: larger than the others, scarred, "
            "wearing crude plate cobbled from human armour, a jagged blade in each hand. "
            "Two bodyguards flank him. And there — tied to a post in the far corner — "
            "is a small boy, Tomas. Alive. Eyes wide. The warlord sees the party "
            "and grins with too many teeth. 4-5 sentences."
        )
        self.narrator(entry_prompt, CONV_ID, label='boss-entry')

        # Brief character reactions before combat starts
        scene_ctx_pre = (
            "The party has entered Snagtooth's chamber. The warlord and two bodyguards stand ready. "
            "Tomas is tied to a post in the far corner — alive, scared. "
            "The fight is about to start."
        )

        prior_pre = []
        text = self.character(
            'kael', scene_ctx_pre,
            "Kael sees Tomas and something shifts in him. "
            "His sword Dawnbreak is glowing bright. He says something — "
            "maybe to Snagtooth, maybe a prayer to Solara. Then he's moving. 2-3 sentences.",
            CONV_ID, label='boss-kael-pre', prior=prior_pre,
        )
        prior_pre.append(('Kael', text))

        self.advance_turn(minutes=1)
        self.character(
            'thorne', scene_ctx_pre,
            "Thorne looks at the chamber — high ceiling, room to move. "
            "Something changes in his bearing after what Kael just said. "
            "He's already deciding to shift. "
            "2-3 sentences — the last words before he's no longer speaking in words.",
            CONV_ID, label='boss-thorne-pre', prior=prior_pre,
        )

        # Run the boss fight — 3 rounds minimum, Kael uses smite, Thorne bears
        rounds = self._run_combat(
            conv_id=CONV_ID,
            heroes=heroes,
            enemies=enemies,
            min_rounds=3,
            thorne_can_bear=True,
            kael_can_smite=True,
            combat_label='boss-fight',
        )

        combat_summary = build_combat_summary(rounds)
        total_rounds = len(rounds)

        # Narrator describes the fight — must include smite moment and bear form
        boss_narration = (
            f"Narrate this entire boss fight as immersive prose. NO game terms at all. "
            f"This is the hardest fight of the adventure — {total_rounds} rounds. "
            f"Key story beats that MUST appear:\n"
            f"1. Kael calls on Solara's power — his sword blazes white-gold as he strikes, "
            f"   the light filling the chamber. It's a moment of genuine divine power.\n"
            f"2. Thorne drops into bear form — describe the transformation, the size, "
            f"   the rage that isn't his. The goblins back away.\n"
            f"3. Snagtooth is a real threat — he hits someone in the party, it matters.\n"
            f"4. Grimble is fighting close, brutal, relentless.\n"
            f"5. Lyra is at range, her 'bardic' magic crackling with violet light "
            f"   she doesn't quite explain.\n"
            f"End with Snagtooth falling. Tomas is still tied to the post, watching. "
            f"6-8 sentences.\n\n"
            f"MECHANICS (reference only):\n{combat_summary}"
        )
        self.narrator(boss_narration, CONV_ID, label='boss-fight-narration')

        self.advance_clock(minutes=20)

    def run_scene_8(self):
        """Scene 8 (conv 107): Tomas — aftermath, cutting him free."""
        CONV_ID = 100
        logger.info("=== Scene 8: Tomas ===")

        aftermath_prompt = (
            "The fight is over. The chamber is quiet except for Tomas's ragged breathing. "
            "The party turns to the boy — he's nine years old, small, "
            "his wrists raw from the bindings. He hasn't cried. He's been trying not to. "
            "The silence after violence is its own texture. 4-5 sentences."
        )
        self.narrator(aftermath_prompt, CONV_ID, label='tomas-open')

        scene_ctx = (
            "The boss fight is over. Tomas — the nine-year-old boy — is tied to a post. "
            "He's scared but unharmed. The party is approaching him. "
            "He doesn't know these people but they just fought to save him."
        )

        # Each character has a moment with Tomas
        char_prompts = {
            'grimble': (
                "Grimble approaches Tomas to check the bindings. "
                "He's looking for a trap — because he'd set one himself. "
                "He finds one: a small pressure mechanism on the rope. He disarms it. "
                "He says something to Tomas — probably short, not tender, "
                "but it's clearly meant to be reassuring. 2-3 sentences."
            ),
            'kael': (
                "Kael kneels down to Tomas's eye level. He's enormous but somehow "
                "manages to make himself small. He says something simple and kind. "
                "Maybe he laughs — his loud relieved laugh. 2-3 sentences."
            ),
            'lyra': (
                "Lyra sings something small — just a few words, almost under her breath. "
                "An old lullaby. Something from before the pact, from when she was a child. "
                "She'd be embarrassed if anyone made a thing of it. 2-3 sentences."
            ),
            'thorne': (
                "Thorne has shifted back to his own form. He crouches down near Tomas — "
                "that particular Firbolg crouch that brings him to ground level. "
                "He says something slow and gentle. Maybe about the forest. 2-3 sentences."
            ),
        }

        prior = []
        for cid in ['grimble', 'kael', 'lyra']:
            text = self.character(
                cid, scene_ctx, char_prompts[cid], CONV_ID,
                label=f'tomas-{cid}', prior=prior,
            )
            prior.append((CHARACTER_DISPLAY_NAMES[cid], text))
            self.advance_turn(minutes=2)

        # Thorne last — sees what the others have done with Tomas
        self.character(
            'thorne', scene_ctx, char_prompts['thorne'], CONV_ID,
            label='tomas-thorne', prior=prior,
        )

        self.advance_clock(minutes=5)

    def run_scene_9(self):
        """Scene 9: Return — walking back through Ashwood at dusk."""
        CONV_ID = 100
        logger.info("=== Scene 9: Return ===")

        return_prompt = (
            "The party is walking back through Ashwood as dusk falls. "
            "Tomas walks between Kael and Thorne. The forest feels different now — "
            "the birds have come back. The light is going orange between the trees. "
            "They can see the road ahead, and at the tree line, a small figure — "
            "Aldric, who has waited all afternoon. 4-5 sentences."
        )
        self.narrator(return_prompt, CONV_ID, label='return-walk')

        walk_ctx = (
            "The party is walking back through Ashwood at dusk. Tomas is with them, safe. "
            "They can see Aldric waiting at the tree line ahead. No one is talking much."
        )

        # One quiet moment on the walk — Grimble, the least sentimental one
        self.advance_turn(minutes=3)
        grimble_walk = self.character(
            'grimble', walk_ctx,
            "Grimble is walking ahead of the others, not looking at anyone. "
            "He says something under his breath — gruff, brief — that reveals more than he intends. "
            "2 sentences.",
            CONV_ID, label='return-walk-grimble',
        )

        # The reunion
        self.advance_turn(minutes=5)
        reunion_prompt = (
            "Aldric sees Tomas at the tree line. Describe the reunion — the old man's face, "
            "the way he moves toward the boy. He's not a demonstrative man. "
            "But everything he's held in for two days comes out in that moment. 4-5 sentences."
        )
        self.narrator(reunion_prompt, CONV_ID, label='return-reunion')

        # Aldric thanks the party
        self.advance_turn(minutes=2)
        aldric_thanks = (
            "Aldric turns to the party. He's trying to hold himself together. "
            "Generate what he says — heartfelt, simple, the words of a farmer who doesn't have "
            "big words for big things. He offers them what he has, which isn't much. "
            "3-4 sentences, first person."
        )
        if self.dry_run:
            aldric_text = narrator_dry('Aldric thanks')
        else:
            sleep_between_calls(4)
            aldric_text = narrator_llm(aldric_thanks)
        self.emit('Aldric', aldric_text, CONV_ID)

        # Farewells — each character gets ONE moment, no doubles.
        # Kael stays and responds to Aldric; Lyra leaves with a small gesture;
        # Thorne is last at the treeline.
        farewell_ctx = (
            "The party is standing at the Ashwood tree line. Aldric has just thanked them. "
            "Tomas is in his grandfather's arms. The road to Thornwall is ahead."
        )
        prior = [('Grimble (earlier on the walk)', grimble_walk)]

        self.advance_turn(minutes=2)
        text = self.character(
            'kael', farewell_ctx,
            "Kael responds to Aldric directly — something warm and simple. "
            "He probably cries a little. He's not embarrassed. 2 sentences.",
            CONV_ID, label='return-kael', prior=prior,
        )
        prior.append(('Kael', text))

        self.advance_turn(minutes=2)
        text = self.character(
            'lyra', farewell_ctx,
            "Lyra's farewell — something small that she leaves behind (a word, a gesture, "
            "a note she slips to Tomas). Charming on the surface but real underneath. "
            "Then she goes. 2 sentences.",
            CONV_ID, label='return-lyra', prior=prior,
        )
        prior.append(('Lyra', text))

        self.advance_turn(minutes=3)
        self.character(
            'thorne', farewell_ctx,
            "Thorne is the last at the tree line. The others are already gone. "
            "He says something — to the forest, or to Tomas, or to no one. "
            "Then he turns and walks toward Thornwall. 2 sentences.",
            CONV_ID, label='return-thorne', prior=prior,
        )

        # Closing narration
        self.advance_turn(minutes=5)
        closing_prompt = (
            "Close the adventure. The road is empty now. Just Aldric and Tomas, "
            "the orange light dying on the Ashwood treeline, the distant torches of Thornwall. "
            "The party is already gone — scattered back into their separate lives. "
            "But something was different tonight. 3-4 sentences. Let it breathe."
        )
        self.narrator(closing_prompt, CONV_ID, label='closing')

        self.advance_clock(minutes=60)

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------

    SCENES = [
        (1, 'Thornwall Market',     'run_scene_1'),
        (2, 'The Plea',             'run_scene_2'),
        (3, 'The Trail',            'run_scene_3'),
        (4, 'The Cave',             'run_scene_4'),
        (5, 'First Blood',          'run_scene_5'),
        (6, 'Through the Den',      'run_scene_6'),
        (7, "Snagtooth's Chamber",  'run_scene_7'),
        (8, 'Tomas',                'run_scene_8'),
        (9, 'Return',               'run_scene_9'),
    ]

    def run(self):
        if not self.skip_wipe and not self.dry_run:
            logger.info("Wiping old adventure data...")
            wipe_old_data()

        print("\n" + "=" * 60)
        print("THE MISSING KIN — A D&D Mini-Adventure")
        print("=" * 60 + "\n")

        for scene_num, scene_name, method_name in self.SCENES:
            if scene_num < self.start_scene:
                logger.info(f"Skipping scene {scene_num}: {scene_name}")
                # Still advance clock for skipped scenes
                skip_advances = {1: 0, 2: 20, 3: 90, 4: 30, 5: 0, 6: 30, 7: 20, 8: 10, 9: 90}
                self.advance_clock(minutes=skip_advances.get(scene_num, 0))
                continue

            print(f"\n{'='*60}")
            print(f"SCENE {scene_num}: {scene_name.upper()}")
            print(f"Time: {self.clock.strftime('%A, %d %B %Y — %H:%M')}")
            print(f"{'='*60}\n")

            method = getattr(self, method_name)
            try:
                method()
            except Exception as e:
                logger.error(f"Scene {scene_num} failed: {e}", exc_info=True)
                if not self.dry_run:
                    raise

        print("\n" + "=" * 60)
        print("Adventure complete.")
        print(f"Final time: {self.clock.strftime('%A, %d %B %Y — %H:%M')}")
        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='The Missing Kin — D&D Mini-Adventure Simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/simulate_adventure.py
  python scripts/simulate_adventure.py --dry-run
  python scripts/simulate_adventure.py --start-scene 5
  python scripts/simulate_adventure.py --skip-wipe --start-scene 7
        """,
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print placeholder narrations without DB writes or LLM calls',
    )
    parser.add_argument(
        '--skip-wipe',
        action='store_true',
        help="Don't delete existing adventure messages before running",
    )
    parser.add_argument(
        '--start-scene',
        type=int,
        default=1,
        choices=range(1, 10),
        metavar='N',
        help='Skip to scene N (1-9), advancing the world clock appropriately',
    )
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] No DB writes. No LLM calls. Placeholder text only.\n")

    runner = AdventureRunner(
        dry_run=args.dry_run,
        skip_wipe=args.skip_wipe,
        start_scene=args.start_scene,
    )
    runner.run()


if __name__ == '__main__':
    main()
