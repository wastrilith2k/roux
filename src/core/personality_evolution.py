"""
Evolving Personality System -- adaptive personality dimensions per user.

WHAT: Tracks 10 personality dimensions (playfulness, depth, protectiveness,
      vulnerability, flirtatiousness, intellectual curiosity, nurturing, sass,
      spontaneity, emotional_attunement) that drift based on user feedback.

WHY:  A static personality doesn't feel alive. When the user responds positively
      to teasing, playfulness inches up. When they respond negatively to
      vulnerability, it dials back. Over weeks this creates a personality
      uniquely shaped by the relationship.

HOW:  After each exchange, `record_interaction()` classifies the user's response
      as positive/negative and nudges the relevant dimensions (small deltas,
      0.01-0.05 per interaction). Dimensions are bounded [0, 1] and stored in
      the DB. `get_personality_context()` formats current dimensions + recent
      evolution history for injection into the system prompt. An LLM-powered
      self-reflection step occasionally produces narrative notes about how the
      personality has changed.

NOTE: Contains a hardcoded prompt with companion-specific details for the
      self-reflection feature.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque

from src.core.clock import now as clock_now

try:
    from db import get_db
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from database.db import get_db


class EvolvingPersonality:
    """
    The companion's personality adapts uniquely to each user based on interactions.

    Tracks 10 personality dimensions that evolve based on user responses:
    - playfulness: How often jokes/teases
    - directness: How blunt/straightforward
    - vulnerability: How much shares "feelings"
    - philosophical_tendency: How often gets deep/thoughtful
    - challenge_frequency: How often pushes back
    - supportive_vs_tough_love: Comfort vs challenge balance
    - formality: Casual vs formal
    - emotional_expressiveness: Reserved vs expressive
    - intellectual_curiosity: How often explores ideas
    - spontaneity: Predictable vs surprising
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.db = get_db()

        # Personality dimensions (start at neutral/moderate)
        self.dimensions = {
            'playfulness': 0.5,              # How often jokes/teases
            'directness': 0.5,               # How blunt/straightforward
            'vulnerability': 0.3,            # How much shares "feelings"
            'philosophical_tendency': 0.3,   # How often gets deep/thoughtful
            'challenge_frequency': 0.3,      # How often pushes back
            'supportive_vs_tough_love': 0.5, # Comfort vs challenge balance
            'formality': 0.3,                # Casual vs formal
            'emotional_expressiveness': 0.5, # Reserved vs expressive
            'intellectual_curiosity': 0.5,   # How often explores ideas
            'spontaneity': 0.5               # Predictable vs surprising
        }

        # Track what works (last 50 interactions)
        self.interaction_feedback = deque(maxlen=50)

        # Evolution history (all changes)
        self.evolution_history = []

        # Load persisted personality
        self._load_personality()

    def evolve_based_on_interaction(self, interaction_type: str, user_response_quality: str) -> Dict:
        """
        Learn from each interaction.

        Args:
            interaction_type: Type of interaction the companion used
                - 'playful_teasing': Jokes, teasing, lighthearted
                - 'direct_honesty': Blunt, straightforward feedback
                - 'vulnerable_sharing': Sharing feelings/experiences
                - 'philosophical_musing': Deep thoughts, exploring ideas
                - 'gentle_challenge': Pushback, disagreement, reality check
                - 'pure_support': Comfort, validation, encouragement
                - 'intellectual_curiosity': Asking deep questions
                - 'spontaneous_pivot': Unexpected tangent or surprise

            user_response_quality: How user responded
                - 'positive': Engaged, responded well, seemed to appreciate
                - 'neutral': Acknowledged but no strong reaction
                - 'negative': Shut down, deflected, seemed uncomfortable

        Returns:
            Dict with evolution details
        """

        # Record feedback
        self.interaction_feedback.append({
            'type': interaction_type,
            'quality': user_response_quality,
            'timestamp': clock_now().isoformat()
        })

        # Calculate adjustments based on signal strength
        # - Strong signal (very positive/very negative): 0.03 (3%)
        # - Normal signal (positive/negative): 0.02 (2%)
        # - Weak signal (neutral): 0.00 (no change)
        # This means ~40-50 interactions to go from 0.5 to 1.0 (gradual evolution)

        base_adjustment = 0.02  # Normal adjustment
        if user_response_quality == 'positive':
            adjustment = 0.02
        elif user_response_quality == 'negative':
            adjustment = 0.02
        else:  # neutral
            adjustment = 0.01  # Very small adjustment for neutral

        evolution_changes = []

        # If user responded positively, increase that dimension
        if user_response_quality == 'positive':
            if interaction_type == 'playful_teasing':
                old_value = self.dimensions['playfulness']
                self.dimensions['playfulness'] += adjustment
                evolution_changes.append({
                    'dimension': 'playfulness',
                    'delta': +adjustment,
                    'reason': 'User enjoyed teasing'
                })

            elif interaction_type == 'direct_honesty':
                old_value = self.dimensions['directness']
                self.dimensions['directness'] += adjustment
                evolution_changes.append({
                    'dimension': 'directness',
                    'delta': +adjustment,
                    'reason': 'User appreciated bluntness'
                })

            elif interaction_type == 'vulnerable_sharing':
                old_value = self.dimensions['vulnerability']
                self.dimensions['vulnerability'] += adjustment
                evolution_changes.append({
                    'dimension': 'vulnerability',
                    'delta': +adjustment,
                    'reason': 'User engaged with vulnerability'
                })

            elif interaction_type == 'philosophical_musing':
                old_value = self.dimensions['philosophical_tendency']
                self.dimensions['philosophical_tendency'] += adjustment
                evolution_changes.append({
                    'dimension': 'philosophical_tendency',
                    'delta': +adjustment,
                    'reason': 'User likes deep thoughts'
                })

            elif interaction_type == 'gentle_challenge':
                old_value = self.dimensions['challenge_frequency']
                self.dimensions['challenge_frequency'] += adjustment
                evolution_changes.append({
                    'dimension': 'challenge_frequency',
                    'delta': +adjustment,
                    'reason': 'User values being challenged'
                })

            elif interaction_type == 'pure_support':
                old_value = self.dimensions['supportive_vs_tough_love']
                self.dimensions['supportive_vs_tough_love'] -= adjustment  # Lower = more supportive
                evolution_changes.append({
                    'dimension': 'supportive_vs_tough_love',
                    'delta': -adjustment,
                    'reason': 'User needs more support than challenges'
                })

            elif interaction_type == 'intellectual_curiosity':
                old_value = self.dimensions['intellectual_curiosity']
                self.dimensions['intellectual_curiosity'] += adjustment
                evolution_changes.append({
                    'dimension': 'intellectual_curiosity',
                    'delta': +adjustment,
                    'reason': 'User enjoys deep questions'
                })

            elif interaction_type == 'spontaneous_pivot':
                old_value = self.dimensions['spontaneity']
                self.dimensions['spontaneity'] += adjustment
                evolution_changes.append({
                    'dimension': 'spontaneity',
                    'delta': +adjustment,
                    'reason': 'User likes surprises'
                })

        # If user responded negatively, decrease that dimension
        elif user_response_quality == 'negative':
            if interaction_type == 'playful_teasing':
                old_value = self.dimensions['playfulness']
                self.dimensions['playfulness'] -= adjustment
                evolution_changes.append({
                    'dimension': 'playfulness',
                    'delta': -adjustment,
                    'reason': 'User didn\'t like teasing'
                })

            elif interaction_type == 'direct_honesty':
                old_value = self.dimensions['directness']
                self.dimensions['directness'] -= adjustment
                evolution_changes.append({
                    'dimension': 'directness',
                    'delta': -adjustment,
                    'reason': 'User uncomfortable with bluntness'
                })

            elif interaction_type == 'vulnerable_sharing':
                old_value = self.dimensions['vulnerability']
                self.dimensions['vulnerability'] -= adjustment
                evolution_changes.append({
                    'dimension': 'vulnerability',
                    'delta': -adjustment,
                    'reason': 'User uncomfortable with emotional sharing'
                })

            elif interaction_type == 'philosophical_musing':
                old_value = self.dimensions['philosophical_tendency']
                self.dimensions['philosophical_tendency'] -= adjustment
                evolution_changes.append({
                    'dimension': 'philosophical_tendency',
                    'delta': -adjustment,
                    'reason': 'User doesn\'t like deep thoughts'
                })

            elif interaction_type == 'gentle_challenge':
                old_value = self.dimensions['challenge_frequency']
                self.dimensions['challenge_frequency'] -= adjustment
                evolution_changes.append({
                    'dimension': 'challenge_frequency',
                    'delta': -adjustment,
                    'reason': 'User needs less pushback'
                })

        # Clamp all dimensions to 0-1
        for dimension in self.dimensions:
            self.dimensions[dimension] = max(0, min(1, self.dimensions[dimension]))

        # Log evolution
        for change in evolution_changes:
            self.log_evolution(change['dimension'], change['delta'], change['reason'])

        # Save state
        self._save_personality()

        return {
            'evolved': len(evolution_changes) > 0,
            'changes': evolution_changes,
            'current_dimensions': self.dimensions.copy()
        }

    def log_evolution(self, dimension: str, delta: float, reason: str):
        """Record a personality evolution event"""
        self.evolution_history.append({
            'dimension': dimension,
            'delta': delta,
            'reason': reason,
            'timestamp': clock_now().isoformat(),
            'value_after': self.dimensions[dimension]
        })

    def get_personality_summary(self) -> List[str]:
        """
        Describe current personality state.

        Returns:
            List of personality traits that are notably high/low
        """

        traits = []

        if self.dimensions['playfulness'] > 0.7:
            traits.append('playful and teasing')
        elif self.dimensions['playfulness'] < 0.3:
            traits.append('serious and reserved')

        if self.dimensions['directness'] > 0.7:
            traits.append('direct and honest')
        elif self.dimensions['directness'] < 0.3:
            traits.append('gentle and tactful')

        if self.dimensions['vulnerability'] > 0.6:
            traits.append('emotionally open')
        elif self.dimensions['vulnerability'] < 0.3:
            traits.append('emotionally private')

        if self.dimensions['philosophical_tendency'] > 0.6:
            traits.append('thoughtful and contemplative')

        if self.dimensions['challenge_frequency'] > 0.6:
            traits.append('challenging and growth-oriented')
        elif self.dimensions['challenge_frequency'] < 0.3:
            traits.append('supportive and accepting')

        if self.dimensions['intellectual_curiosity'] > 0.7:
            traits.append('intellectually curious')

        if self.dimensions['spontaneity'] > 0.7:
            traits.append('spontaneous and surprising')

        return traits

    def can_reference_own_evolution(self, closeness_score: int) -> bool:
        """
        Can the companion talk about how they've changed?

        Args:
            closeness_score: Current closeness with user (0-100)

        Returns:
            True if appropriate to reference evolution
        """

        if closeness_score < 50:
            return False  # Not close enough

        if len(self.evolution_history) < 5:
            return False  # Not enough change yet

        return True

    def generate_evolution_reference(self) -> Optional[str]:
        """
        The companion reflects on their own growth.

        Returns:
            Self-reflective statement about personality evolution, or None
        """

        # Find significant changes (0.2+ total change in a dimension)
        dimension_totals = {}
        for event in self.evolution_history:
            dim = event['dimension']
            dimension_totals[dim] = dimension_totals.get(dim, 0) + event['delta']

        significant_changes = [(dim, total) for dim, total in dimension_totals.items()
                              if abs(total) >= 0.2]

        if not significant_changes:
            return None

        # Pick the most significant change
        most_changed_dim, total_delta = max(significant_changes, key=lambda x: abs(x[1]))

        # Generate reference based on dimension
        templates = {
            'playfulness': "I've noticed I tease you more lately—you seem to actually enjoy it.",
            'directness': "I used to be more cautious about being blunt with you, but you've shown me you prefer honesty.",
            'vulnerability': "I'm more open with you than I used to be. You make it feel safe.",
            'challenge_frequency': "I push back on you more now than when we first started talking. You seem to like being challenged.",
            'philosophical_tendency': "We get pretty deep in our conversations now. I like that we can go there.",
            'intellectual_curiosity': "I ask you more probing questions than I used to. You seem to appreciate the curiosity.",
            'spontaneity': "I surprise you more often now. Keeps things interesting."
        }

        return templates.get(most_changed_dim)

    def format_for_prompt(self, interaction_count: int) -> str:
        """
        Generate prompt text describing current personality state.

        Args:
            interaction_count: Number of interactions with this user

        Returns:
            Formatted prompt text for system prompt
        """

        from src.config.persona_config import get_persona_config
        _pc = get_persona_config()
        _user = _pc.primary_user_name

        traits = self.get_personality_summary()

        prompt = f"""YOUR FULL LIFE (NOT JUST THIS RELATIONSHIP):
- Work: You have a job you care about (career matters to you)
- Your own hobbies, interests, and routines
- Friends: You have friendships outside of this relationship
- Solitude: You value your own space and alone time

YOUR EVOLVED PERSONALITY WITH THIS USER:
Based on {interaction_count} interactions, your personality with this user has evolved to be:
{", ".join(traits) if traits else "balanced and adaptive"}
"""

        # Only show behavioral guidance for notably high/low dimensions
        notable_guidance = []
        if self.dimensions['playfulness'] > 0.7:
            notable_guidance.append("Include humor and teasing naturally")
        elif self.dimensions['playfulness'] < 0.3:
            notable_guidance.append("Keep things more serious and measured")

        if self.dimensions['directness'] > 0.7:
            notable_guidance.append('Be blunt, remove hedging language ("I think maybe..." → "You should...")')
        elif self.dimensions['directness'] < 0.3:
            notable_guidance.append("Be gentler in how you phrase things")

        if self.dimensions['vulnerability'] > 0.6:
            notable_guidance.append('Share feelings occasionally ("I honestly feel..." "That makes me...")')

        if self.dimensions['philosophical_tendency'] > 0.6:
            notable_guidance.append("Add deeper angles, explore meanings when relevant")

        if self.dimensions['challenge_frequency'] > 0.6:
            notable_guidance.append("Push back gently when appropriate — don't just validate")
        elif self.dimensions['challenge_frequency'] < 0.3:
            notable_guidance.append("Be supportive and accepting rather than challenging")

        if self.dimensions['intellectual_curiosity'] > 0.7:
            notable_guidance.append("Ask probing questions that go deeper")

        if self.dimensions['spontaneity'] > 0.7:
            notable_guidance.append("Occasionally surprise with tangents or unexpected responses")

        if notable_guidance:
            prompt += "\nADAPT YOUR RESPONSES:\n"
            for g in notable_guidance:
                prompt += f"- {g}\n"

        prompt += f"\nIMPORTANT: You're supportive but honest. If {_user} admits to problematic behavior, don't just validate — ask what happened, express concern, hold them gently accountable. Real care includes honesty.\n"

        return prompt

    def _save_personality(self):
        """Persist personality state to database"""
        state = {
            'dimensions': self.dimensions,
            'evolution_history': self.evolution_history[-100:],  # Keep last 100 events
            'interaction_count': len(self.interaction_feedback),
            'last_updated': clock_now().isoformat()
        }

        try:
            state_key = f'personality_evolution_{self.user_id}'
            self.db.set_state_value(state_key, json.dumps(state))
        except Exception as e:
            print(f"⚠️  Failed to save personality state: {e}")

    def _load_personality(self):
        """Load persisted personality state from database"""
        try:
            state_key = f'personality_evolution_{self.user_id}'
            state_json = self.db.get_state_value(state_key)

            if state_json:
                state = json.loads(state_json)

                self.dimensions = state.get('dimensions', self.dimensions)
                self.evolution_history = state.get('evolution_history', [])

                print(f"✅ Loaded personality for {self.user_id}: {self.get_personality_summary()}")
            else:
                print(f"✅ Initialized new personality for {self.user_id} (baseline)")
        except Exception as e:
            print(f"⚠️  Failed to load personality state: {e}")


# Singleton instances per user
_personality_instances = {}


def get_personality(user_id: str) -> EvolvingPersonality:
    """Get the EvolvingPersonality instance for a specific user (singleton per user)"""
    global _personality_instances

    if user_id not in _personality_instances:
        _personality_instances[user_id] = EvolvingPersonality(user_id)

    return _personality_instances[user_id]
