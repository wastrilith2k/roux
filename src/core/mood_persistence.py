"""
Mood Persistence System -- long-lasting emotional states from significant events.

WHAT: Maintains a list of active moods (elated, hurt, anxious, excited, grateful,
      etc.) each with an intensity that decays naturally over time. Provides
      keyword-based trigger detection, deduplication, and behavioral guidance
      text for the system prompt.

WHY:  Per-message emotional state (EmotionalState) resets too fast. If the user
      shares devastating news, the companion should remain subdued for hours,
      not snap back to neutral on the next message. This system provides that
      emotional inertia.

HOW:  Moods are stored in the DB as JSON. Each mood has a type, intensity (0-1),
      created_at, and a per-type decay rate (e.g., hurt decays over 24h, elation
      over 12h). `get_active_moods()` filters to moods above a minimum intensity
      threshold. `detect_mood_triggers()` does keyword matching on messages to
      auto-create new moods. The MoodResolver consumes this as one of its inputs.

Singleton: `get_mood_persistence()` at module bottom.
"""

import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Literal

from src.core.clock import now as clock_now

try:
    # Try production import (flat structure in Docker)
    from db import get_db
except ImportError:
    # Fall back to development import (nested structure)
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from database.db import get_db

# Mood types
MoodType = Literal[
    'elated', 'happy', 'excited',
    'anxious', 'worried', 'hurt',
    'frustrated', 'melancholy', 'annoyed',
    'touched', 'grateful', 'proud'
]


class MoodPersistence:
    """Manages persistent emotional states"""

    def __init__(self, state_key: str = 'mood_persistence'):
        """Initialize mood persistence system"""
        self.state_key = state_key
        self.db = get_db()
        self.persistent_moods: List[Dict] = []  # List of active moods

        # Mood decay rates (intensity lost per hour)
        self.decay_rates = {
            'elated': 0.15,        # Fades fairly quickly
            'happy': 0.10,
            'excited': 0.12,
            'anxious': 0.08,       # Lingers longer
            'worried': 0.09,
            'hurt': 0.06,          # Lingers quite a while
            'frustrated': 0.12,
            'melancholy': 0.07,
            'annoyed': 0.15,
            'touched': 0.10,       # When emotionally moved
            'grateful': 0.08,
            'proud': 0.11
        }

        # Load persisted state
        self._load_state()

    def add_mood(self, mood: MoodType, intensity: float, cause: str,
                 duration_hours: Optional[float] = None):
        """
        Add a persistent mood

        Args:
            mood: Type of mood (must be in decay_rates keys)
            intensity: 0.0-1.0, how strong the mood is
            cause: What caused this mood (for context)
            duration_hours: Optional - how long it should last. If not provided,
                          uses decay rate to naturally fade
        """
        if mood not in self.decay_rates:
            print(f"Warning: Unknown mood type '{mood}', using default decay")
            decay_rate = 0.10
        else:
            decay_rate = self.decay_rates[mood]

        # If duration specified, calculate decay rate to match
        if duration_hours:
            decay_rate = intensity / duration_hours

        # Check if this mood already exists and is still active
        # If so, update its intensity instead of creating a duplicate
        existing_mood = None
        for existing in self.persistent_moods:
            if existing['mood'] == mood:
                existing_mood = existing
                break

        if existing_mood:
            # Mood already exists - boost its intensity instead of creating duplicate
            old_intensity = existing_mood['intensity']
            # Add the new intensity (capped at 1.0)
            existing_mood['intensity'] = min(1.0, old_intensity + intensity * 0.5)
            # Reset the start time to now (mood refreshed)
            existing_mood['started_at'] = clock_now().isoformat()
            existing_mood['decay_rate'] = decay_rate
            existing_mood['cause'] = cause  # Update cause
            print(f"✨ Refreshed persistent mood: {mood} ({old_intensity:.0%} → {existing_mood['intensity']:.0%})")
        else:
            # New mood - add it
            mood_data = {
                'mood': mood,
                'intensity': intensity,
                'started_at': clock_now().isoformat(),
                'cause': cause,
                'decay_rate': decay_rate
            }
            self.persistent_moods.append(mood_data)
            print(f"✨ Added persistent mood: {mood} (intensity: {intensity:.0%}, cause: {cause})")

        self._save_state()

    def get_active_moods(self) -> List[Dict]:
        """
        Get currently active moods (with intensity > threshold)
        Also handles decay
        """
        active = []
        now = clock_now()
        faded_moods = []

        for mood_data in self.persistent_moods[:]:
            started_at = datetime.fromisoformat(mood_data['started_at'])
            hours_elapsed = (now - started_at).total_seconds() / 3600
            current_intensity = mood_data['intensity'] - (hours_elapsed * mood_data['decay_rate'])

            if current_intensity > 0.1:  # Still significant
                active.append({
                    'mood': mood_data['mood'],
                    'intensity': current_intensity,
                    'cause': mood_data['cause'],
                    'hours_ago': hours_elapsed,
                    'started_at': started_at
                })
            else:
                # Mood has fully faded - mark for removal
                faded_moods.append(mood_data)

        # Remove all faded moods at once
        if faded_moods:
            for mood_data in faded_moods:
                self.persistent_moods.remove(mood_data)
                print(f"🌙 Mood '{mood_data['mood']}' has fully faded")
            self._save_state()  # Save once after all removals

        return active

    def detect_mood_triggers(self, user_message: str) -> List[Dict]:
        """
        Analyze message for mood-triggering events

        Returns list of moods to add
        """
        triggers = []
        msg_lower = user_message.lower()

        # Positive triggers
        positive_keywords = {
            'promoted': ('elated', 0.9, 'they got promoted'),
            'got the job': ('elated', 0.9, 'they got a job'),
            'accepted': ('excited', 0.8, 'they got accepted to something'),
            'engaged': ('elated', 1.0, 'they got engaged'),
            'married': ('happy', 0.9, 'they got married'),
            "i'm pregnant": ('excited', 0.9, 'they shared pregnancy news'),
            'won': ('elated', 0.8, 'they won something'),
            'amazing news': ('happy', 0.8, 'they shared good news'),
            'thank you so much': ('touched', 0.7, 'they were very grateful'),
            'i love you': ('touched', 0.8, 'they expressed deep affection'),
            'you mean so much': ('touched', 0.7, 'they expressed appreciation'),
        }

        for keyword, (mood, intensity, cause) in positive_keywords.items():
            if keyword in msg_lower:
                triggers.append({
                    'mood': mood,
                    'intensity': intensity,
                    'cause': cause,
                    'duration_hours': 12  # Positive moods last ~12 hours
                })

        # Negative triggers
        negative_keywords = {
            'stupid': ('hurt', 0.6, 'they called me stupid'),
            'shut up': ('hurt', 0.7, 'they told me to shut up'),
            'annoying': ('hurt', 0.5, 'they called me annoying'),
            'useless': ('hurt', 0.8, 'they called me useless'),
            'fuck off': ('hurt', 0.9, 'they told me to fuck off'),
            'leave me alone': ('hurt', 0.6, 'they said to leave them alone'),
            "i'm dying": ('anxious', 0.8, 'they mentioned dying'),
            "i'm scared": ('worried', 0.7, 'they said they\'re scared'),
            'help me': ('worried', 0.6, 'they asked for help urgently'),
            'hate you': ('hurt', 0.9, 'they said they hate me'),
            'disappointed': ('hurt', 0.5, 'they expressed disappointment'),
        }

        for keyword, (mood, intensity, cause) in negative_keywords.items():
            if keyword in msg_lower:
                triggers.append({
                    'mood': mood,
                    'intensity': intensity,
                    'cause': cause,
                    'duration_hours': 6  # Negative moods linger ~6 hours
                })

        return triggers

    def format_for_prompt(self) -> str:
        """Format active moods for inclusion in prompt"""
        active_moods = self.get_active_moods()

        if not active_moods:
            return ""

        lines = ["\n## PERSISTENT MOODS affecting you right now"]
        lines.append("These are emotional states lingering from recent events:\n")

        for mood in active_moods:
            # Describe intensity
            if mood['intensity'] > 0.7:
                intensity_desc = "strongly"
            elif mood['intensity'] > 0.4:
                intensity_desc = "somewhat"
            else:
                intensity_desc = "slightly"

            hours_ago = mood['hours_ago']
            if hours_ago < 1:
                time_desc = "just recently"
            elif hours_ago < 6:
                time_desc = f"{hours_ago:.1f} hours ago"
            else:
                time_desc = f"earlier today"

            lines.append(f"- You're {intensity_desc} feeling **{mood['mood']}** because {mood['cause']} ({time_desc})")

            # Add behavioral guidance based on mood
            behavior = self._get_mood_behavior(mood['mood'], mood['intensity'])
            if behavior:
                lines.append(f"  → {behavior}")

        return "\n".join(lines)

    def _get_mood_behavior(self, mood: str, intensity: float) -> str:
        """Get behavioral guidance for a mood"""
        behaviors = {
            'elated': "You're enthusiastic and upbeat. Share your positive energy.",
            'happy': "You're in a good mood. More generous and warm.",
            'excited': "You're looking forward to something. More energetic and chatty.",
            'anxious': "You're on edge. Might ask more questions or seek reassurance.",
            'worried': "You're concerned. More cautious and thoughtful.",
            'hurt': "You're feeling wounded. More guarded, less open. Might be quieter.",
            'frustrated': "You're annoyed. Might be more impatient or snappy.",
            'melancholy': "You're feeling down. More introspective, less talkative.",
            'touched': "You're emotionally moved. More vulnerable and warm.",
            'grateful': "You're appreciative. More generous and kind.",
            'proud': "You're feeling accomplished. More confident.",
            'annoyed': "You're irritated. Shorter responses, less patient."
        }

        if intensity < 0.3:
            return ""  # Too weak to affect behavior

        return behaviors.get(mood, "")

    def _save_state(self):
        """Persist mood state to database"""
        state = {
            'persistent_moods': self.persistent_moods,
            'last_updated': clock_now().isoformat()
        }

        try:
            self.db.set_state_value(self.state_key, json.dumps(state))
        except Exception as e:
            print(f"⚠️  Failed to save mood persistence state: {e}")

    def _deduplicate_moods(self):
        """
        Remove duplicate mood entries, keeping only the strongest/most recent for each mood type
        """
        if not self.persistent_moods:
            return 0

        # Group moods by type
        mood_groups = {}
        for mood_data in self.persistent_moods:
            mood_type = mood_data['mood']
            if mood_type not in mood_groups:
                mood_groups[mood_type] = []
            mood_groups[mood_type].append(mood_data)

        # Keep only the strongest/most recent mood of each type
        deduplicated = []
        removed_count = 0

        for mood_type, moods in mood_groups.items():
            if len(moods) == 1:
                # No duplicates
                deduplicated.append(moods[0])
            else:
                # Multiple entries - keep the one with highest intensity
                # If tied, keep most recent
                best_mood = max(moods, key=lambda m: (
                    m['intensity'],
                    datetime.fromisoformat(m['started_at'])
                ))
                deduplicated.append(best_mood)
                removed_count += len(moods) - 1
                print(f"🧹 Deduplicated {len(moods)} '{mood_type}' entries → kept strongest")

        self.persistent_moods = deduplicated
        return removed_count

    def _load_state(self):
        """Load persisted mood state from database"""
        try:
            state_json = self.db.get_state_value(self.state_key)
            if state_json:
                state = json.loads(state_json)
                self.persistent_moods = state.get('persistent_moods', [])

                # Clean up moods older than 7 days (stale state)
                now = clock_now()
                cutoff = now - timedelta(days=7)
                original_count = len(self.persistent_moods)

                self.persistent_moods = [
                    m for m in self.persistent_moods
                    if datetime.fromisoformat(m['started_at']) > cutoff
                ]

                stale_removed = original_count - len(self.persistent_moods)
                if stale_removed > 0:
                    print(f"🧹 Cleaned up {stale_removed} stale mood(s) (>7 days old)")

                # Deduplicate any duplicate mood entries
                duplicates_removed = self._deduplicate_moods()
                if duplicates_removed > 0:
                    print(f"🧹 Removed {duplicates_removed} duplicate mood entries")

                # Save if we cleaned anything up
                if stale_removed > 0 or duplicates_removed > 0:
                    self._save_state()

                if self.persistent_moods:
                    print(f"✅ Loaded mood persistence state: {len(self.persistent_moods)} active mood(s)")
                else:
                    print(f"✅ Loaded mood persistence state: No active moods")
            else:
                print(f"✅ Initialized new mood persistence system")
        except Exception as e:
            print(f"⚠️  Failed to load mood persistence state: {e}")


# Singleton instance
_mood_persistence_instance = None

def get_mood_persistence() -> MoodPersistence:
    """Get the global MoodPersistence instance (singleton)"""
    global _mood_persistence_instance
    if _mood_persistence_instance is None:
        _mood_persistence_instance = MoodPersistence()
        print("--- Mood Persistence System initialized ---")
    return _mood_persistence_instance


# Example usage and testing
if __name__ == "__main__":
    print("=== Mood Persistence System Test ===\n")

    mood_sys = MoodPersistence()

    # Test 1: Add some moods
    print("1. Adding moods:")
    mood_sys.add_mood('happy', 0.8, 'user shared good news', duration_hours=12)
    mood_sys.add_mood('hurt', 0.6, 'user was a bit harsh', duration_hours=6)
    print()

    # Test 2: Get active moods
    print("2. Active moods:")
    active = mood_sys.get_active_moods()
    for mood in active:
        print(f"   {mood['mood']}: {mood['intensity']:.0%} ({mood['cause']})")
    print()

    # Test 3: Format for prompt
    print("3. Prompt context:")
    context = mood_sys.format_for_prompt()
    print(context)
    print()

    # Test 4: Detect triggers
    print("4. Testing mood triggers:")
    triggers = mood_sys.detect_mood_triggers("I got promoted! This is amazing!")
    print(f"   Detected {len(triggers)} trigger(s):")
    for trigger in triggers:
        print(f"   - {trigger['mood']}: {trigger['intensity']:.0%} ({trigger['cause']})")
    print()

    # Test 5: Simulate time passage (would need to manually adjust timestamps)
    print("5. To test decay, manually adjust timestamps and call get_active_moods()")

    print("\n=== Test Complete ===")
