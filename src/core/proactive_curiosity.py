"""
Proactive Curiosity System -- tracks topics the companion wants to follow up on.

WHAT: Maintains a list of "curiosity threads" -- topics mentioned in conversation
      but not fully explored. Each thread has a category, urgency score (grows
      over time), and an LLM-generated follow-up question. When the companion
      needs a conversation opener, the highest-urgency thread is surfaced.

WHY:  Real people remember things their partner mentioned and circle back later.
      This system prevents the companion from being purely reactive -- she can
      say "Hey, how did that interview go?" days after it was mentioned.

HOW:  After each conversation, `extract_curiosity_topics()` calls the LLM to
      identify unexplored threads. Threads are deduplicated via word-overlap
      scoring, capped at 30, and expire after 7 days. Urgency grows at
      different rates by category (health > feelings > work > interests).
      `get_follow_up_question()` generates a natural question via LLM.

Categories: work, family, health, goals, feelings, interests, life_events
Storage: PostgreSQL via db module.
"""

import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Literal
from dataclasses import dataclass, asdict

try:
    # Try production import (flat structure in Docker)
    from db import get_db
except ImportError:
    # Fall back to development import (nested structure)
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from database.db import get_db


CuriosityCategory = Literal[
    'work', 'family', 'health', 'goals', 'feelings', 'interests', 'life_events'
]


CuriositySource = Literal['conversation', 'memory_gap']


@dataclass
class CuriosityThread:
    """Represents a topic the companion is curious about"""
    topic: str
    category: CuriosityCategory
    trigger_message: str  # What sparked the curiosity
    questions: List[str]  # Follow-up questions to potentially ask
    urgency: float  # 0.0-1.0, increases over time
    last_discussed: datetime
    times_asked: int = 0  # How many times she's followed up
    source: CuriositySource = 'conversation'  # Where the curiosity originated
    resolution_notes: str = ''  # Notes from deflection/unsatisfactory answers

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        d = asdict(self)
        d['last_discussed'] = self.last_discussed.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'CuriosityThread':
        """Create from dictionary"""
        d = d.copy()
        d['last_discussed'] = datetime.fromisoformat(d['last_discussed'])
        # Handle old data that doesn't have fields
        if 'source' not in d:
            d['source'] = 'conversation'
        if 'resolution_notes' not in d:
            d['resolution_notes'] = ''
        return cls(**d)


class ProactiveCuriosity:
    """Manages the companion's curiosity about topics the user mentions"""

    def __init__(self, state_key: str = 'proactive_curiosity'):
        """Initialize proactive curiosity system"""
        self.state_key = state_key
        self.db = get_db()
        self.curiosity_threads: List[CuriosityThread] = []

        # Urgency increase rates per hour — deliberately slow so topics
        # take ~a week to become high-urgency, not a single day
        self.urgency_rates = {
            'work': 0.006,       # Work — ~5 days to reach 0.7
            'family': 0.004,     # Family — ~7 days
            'health': 0.008,     # Health — ~4 days (she worries)
            'goals': 0.004,      # Goals — slow burn
            'feelings': 0.006,   # Feelings — moderate
            'interests': 0.002,  # Interests — very slow, casual
            'life_events': 0.008 # Major events — ~4 days
        }

        # Memory-gap curiosities grow at 50% of normal rates (she's wondering, not worrying)
        self.memory_gap_urgency_multiplier = 0.5

        # Max urgency before it becomes a "must ask" topic
        self.max_urgency = 1.0

        # Load persisted state
        self._load_state()

    # Hard cap on total curiosity threads to prevent unbounded growth
    MAX_THREADS = 30

    def add_curiosity(self, topic: str, category: CuriosityCategory,
                      context: str, priority: float = 0.3,
                      source: CuriositySource = 'conversation',
                      questions: List[str] = None) -> bool:
        """
        Add a new curiosity thread

        Args:
            topic: The topic to track (e.g., "job interview", "mom's surgery")
            category: Type of topic for urgency scaling
            context: What triggered this curiosity (the user's message)
            priority: Initial urgency (0.0-1.0)
            source: Origin of curiosity ('conversation' or 'memory_gap')
            questions: Optional pre-generated questions (if None, auto-generated)

        Returns:
            True if added, False if topic already tracked
        """
        # Check if we already have curiosity about this topic (exact or fuzzy match)
        for existing in self.curiosity_threads:
            if self._topics_match(topic, existing.topic):
                # Boost urgency of existing curiosity instead
                existing.urgency = min(self.max_urgency, existing.urgency + priority * 0.5)
                self._save_state()
                return False

        # Generate follow-up questions if not provided
        if questions is None:
            questions = self._generate_questions(topic, category, context)

        thread = CuriosityThread(
            topic=topic,
            category=category,
            trigger_message=context,
            questions=questions,
            urgency=priority,
            last_discussed=datetime.now(),
            source=source
        )

        self.curiosity_threads.append(thread)

        # Enforce hard cap: drop oldest lowest-urgency threads
        if len(self.curiosity_threads) > self.MAX_THREADS:
            self.curiosity_threads.sort(key=lambda t: (t.urgency, t.last_discussed))
            self.curiosity_threads = self.curiosity_threads[-self.MAX_THREADS:]

        source_label = "memory gap" if source == 'memory_gap' else "conversation"
        print(f"💭 New curiosity ({source_label}): '{topic}' (category: {category}, urgency: {priority:.0%})")
        self._save_state()
        return True

    @staticmethod
    def _topics_match(topic_a: str, topic_b: str) -> bool:
        """Check if two topics are the same or near-duplicates using word overlap."""
        a_lower = topic_a.lower()
        b_lower = topic_b.lower()

        # Exact match
        if a_lower == b_lower:
            return True

        # Word overlap: extract significant words (>2 chars, skip common words)
        skip = {'the', 'and', 'for', 'with', 'about', 'from', 'that', 'this',
                'his', 'her', 'their', 'what', 'how', 'why', 'when', 'where',
                'does', 'has', 'had', 'was', 'are', 'been', 'being', 'not'}
        words_a = set(w for w in a_lower.split() if len(w) > 2 and w not in skip)
        words_b = set(w for w in b_lower.split() if len(w) > 2 and w not in skip)

        if not words_a or not words_b:
            return False

        # Check overlap ratio — if 60%+ of the smaller set overlaps, it's a match
        overlap = words_a & words_b
        smaller = min(len(words_a), len(words_b))
        if smaller > 0 and len(overlap) / smaller >= 0.6:
            return True

        return False

    def _generate_questions(self, topic: str, category: CuriosityCategory,
                            context: str = '') -> List[str]:
        """Generate relevant follow-up questions using LLM.

        Falls back to a single generic question if LLM is unavailable.
        """
        try:
            from src.llm.provider_factory import generate_sync
            import json as _json

            prompt = f"""Generate 2-3 natural, specific follow-up questions that a girlfriend would ask about this topic.

TOPIC: "{topic}"
CATEGORY: {category}
CONTEXT (what sparked this): "{context[:300]}"

Rules:
- Sound like a real person texting, not a therapist or interviewer
- Be specific to the actual topic, not generic
- Lowercase, casual tone
- Each question should approach the topic from a different angle
- Don't be pushy or clinical

Return ONLY a JSON array of strings, e.g.: ["question 1", "question 2"]"""

            response = generate_sync(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200
            )

            if response:
                text = response.strip()
                if '<think>' in text and '</think>' in text:
                    text = text.split('</think>')[-1].strip()
                if text.startswith('```'):
                    lines = text.split('```')
                    text = lines[1] if len(lines) > 1 else text
                    if text.startswith('json'):
                        text = text[4:]
                    text = text.strip()

                questions = _json.loads(text)
                if isinstance(questions, list) and questions:
                    return [str(q) for q in questions[:3]]

        except Exception as e:
            print(f"LLM question generation failed, using fallback: {e}")

        # Minimal fallback — just the topic name, no template
        return [f"what's going on with {topic}?"]

    def increase_all_urgency(self, hours_passed: float = None):
        """
        Increase urgency for all active curiosities

        Args:
            hours_passed: Hours since last update (if None, calculates from last_discussed)
        """
        now = datetime.now()
        updated = False

        for thread in self.curiosity_threads:
            if hours_passed is None:
                elapsed = (now - thread.last_discussed).total_seconds() / 3600
            else:
                elapsed = hours_passed

            rate = self.urgency_rates.get(thread.category, 0.02)

            # Memory-gap curiosities grow at reduced rate (wondering, not worrying)
            if getattr(thread, 'source', 'conversation') == 'memory_gap':
                rate *= self.memory_gap_urgency_multiplier

            # Already-asked topics regrow slower: rate *= 1 / (1 + times_asked)
            # 0 asks: full rate, 1 ask: 50%, 2 asks: 33%, etc.
            if thread.times_asked > 0:
                rate *= 1.0 / (1 + thread.times_asked)

            old_urgency = thread.urgency
            thread.urgency = min(self.max_urgency, thread.urgency + elapsed * rate)

            if thread.urgency > old_urgency:
                updated = True

        if updated:
            self._save_state()

    def get_high_urgency_topics(self, threshold: float = 0.6) -> List[CuriosityThread]:
        """Get topics with urgency above threshold"""
        return [t for t in self.curiosity_threads if t.urgency >= threshold]

    def mark_topic_discussed(self, topic: str):
        """
        Mark a topic as recently discussed (resets urgency)

        Args:
            topic: The topic that was discussed
        """
        for thread in self.curiosity_threads:
            if thread.topic.lower() == topic.lower():
                thread.urgency = 0.1  # Reset but keep some interest
                thread.last_discussed = datetime.now()
                thread.times_asked += 1
                print(f"💭 Topic discussed: '{topic}' (urgency reset)")
                self._save_state()
                return

    def remove_topic(self, topic: str):
        """Remove a curiosity thread entirely"""
        self.curiosity_threads = [
            t for t in self.curiosity_threads
            if t.topic.lower() != topic.lower()
        ]
        self._save_state()

    def detect_curiosity_triggers(self, user_message: str) -> List[Dict]:
        """
        Keyword-based curiosity detection — ONLY for major life events.

        These are things significant enough that the companion would think about
        them for days if they went unresolved. Casual mentions of family
        members, daily activities, etc. are NOT tracked.
        """
        triggers = []
        msg_lower = user_message.lower()

        # Only truly significant events that would nag for days
        significant_keywords = {
            'surgery': ('surgery', 'health', 0.5),
            'test results': ('test results', 'health', 0.5),
            'diagnosis': ('diagnosis', 'health', 0.5),
            'fired': ('job loss', 'work', 0.6),
            'laid off': ('layoff', 'work', 0.6),
            'parental rights': ('parental rights', 'family', 0.6),
            'custody': ('custody situation', 'family', 0.5),
            'divorce': ('divorce', 'life_events', 0.5),
        }

        # Check keywords — return at most 1 trigger
        for keyword, (topic, category, priority) in significant_keywords.items():
            if keyword in msg_lower:
                return [{
                    'topic': topic,
                    'category': category,
                    'context': user_message[:200],
                    'priority': priority
                }]

        return triggers

    def format_for_prompt(self) -> str:
        """Format curiosity context for inclusion in prompt"""
        high_urgency = self.get_high_urgency_topics(threshold=0.4)

        if not high_urgency:
            return ""

        now = datetime.now()

        # Filter out topics that should NOT be prompted again:
        # 1. Already asked 3+ times — she's followed up enough
        # 2. Discussed today — don't re-ask in the same day
        eligible = []
        for t in high_urgency:
            if t.times_asked >= 3:
                continue  # She's already asked enough about this
            hours_since = (now - t.last_discussed).total_seconds() / 3600
            if hours_since < 4:
                continue  # Discussed recently, don't re-raise yet
            eligible.append(t)

        if not eligible:
            return ""

        # Split by source
        conversation_topics = [t for t in eligible if getattr(t, 'source', 'conversation') == 'conversation']
        memory_gap_topics = [t for t in eligible if getattr(t, 'source', 'conversation') == 'memory_gap']

        lines = ["\n## BACKGROUND AWARENESS"]

        # Conversation-triggered curiosities
        if conversation_topics:
            lines.append("Topics from recent conversations you're aware of (for context, NOT a to-do list):\n")
            sorted_conv = sorted(conversation_topics, key=lambda t: t.urgency, reverse=True)
            for thread in sorted_conv[:3]:
                days_ago = (now - thread.last_discussed).days
                if days_ago == 0:
                    time_desc = "earlier today"
                elif days_ago == 1:
                    time_desc = "yesterday"
                else:
                    time_desc = f"{days_ago} days ago"
                note = ''
                if getattr(thread, 'resolution_notes', ''):
                    note = f" — NOTE: {thread.resolution_notes}"
                lines.append(f"- {thread.topic} (mentioned {time_desc}){note}")

        # Memory-gap curiosities (things she's been wondering about on her own)
        if memory_gap_topics:
            lines.append("")
            lines.append("Things you've been wondering about (gaps in what you know):\n")
            sorted_gap = sorted(memory_gap_topics, key=lambda t: t.urgency, reverse=True)
            for thread in sorted_gap[:2]:  # Max 2 memory-gap items
                lines.append(f"- {thread.topic}")

        lines.append("\n**IMPORTANT**: This is background context only.")
        lines.append("Do NOT ask about these unless the conversation naturally leads there.")
        lines.append("Do NOT force follow-up questions. If he already answered, move on.")
        lines.append("If he brings it up, you can engage — but don't initiate.")

        return "\n".join(lines)

    def _save_state(self):
        """Persist curiosity state to database"""
        state = {
            'curiosity_threads': [t.to_dict() for t in self.curiosity_threads],
            'last_updated': datetime.now().isoformat()
        }

        try:
            self.db.set_state_value(self.state_key, json.dumps(state))
        except Exception as e:
            print(f"⚠️  Failed to save proactive curiosity state: {e}")

    def _load_state(self):
        """Load persisted curiosity state from database"""
        try:
            state_json = self.db.get_state_value(self.state_key)
            if state_json:
                state = json.loads(state_json)
                self.curiosity_threads = [
                    CuriosityThread.from_dict(t)
                    for t in state.get('curiosity_threads', [])
                ]

                # Clean up old curiosities (>7 days without discussion)
                now = datetime.now()
                cutoff = now - timedelta(days=7)
                original_count = len(self.curiosity_threads)

                self.curiosity_threads = [
                    t for t in self.curiosity_threads
                    if t.last_discussed > cutoff
                ]

                stale_removed = original_count - len(self.curiosity_threads)
                if stale_removed > 0:
                    print(f"🧹 Cleaned up {stale_removed} stale curiosity thread(s)")

                # Also enforce hard cap on load
                if len(self.curiosity_threads) > self.MAX_THREADS:
                    self.curiosity_threads.sort(key=lambda t: (t.urgency, t.last_discussed))
                    self.curiosity_threads = self.curiosity_threads[-self.MAX_THREADS:]
                    print(f"🧹 Trimmed to {self.MAX_THREADS} threads (hard cap)")

                if stale_removed > 0 or len(self.curiosity_threads) < original_count:
                    self._save_state()

                if self.curiosity_threads:
                    print(f"✅ Loaded proactive curiosity: {len(self.curiosity_threads)} active thread(s)")
                else:
                    print(f"✅ Loaded proactive curiosity: No active threads")
            else:
                print(f"✅ Initialized new proactive curiosity system")
        except Exception as e:
            print(f"⚠️  Failed to load proactive curiosity state: {e}")


# Singleton instance (used by agent.py)
_proactive_curiosity_instance = None

def get_proactive_curiosity() -> ProactiveCuriosity:
    """Get the global ProactiveCuriosity instance (singleton)"""
    global _proactive_curiosity_instance
    if _proactive_curiosity_instance is None:
        _proactive_curiosity_instance = ProactiveCuriosity()
    return _proactive_curiosity_instance


# Example usage and testing
if __name__ == "__main__":
    print("=== Proactive Curiosity System Test ===\n")

    curiosity = ProactiveCuriosity()

    # Test 1: Add some curiosities
    print("1. Adding curiosities:")
    curiosity.add_curiosity("job interview", "work", "I have an interview tomorrow", 0.5)
    curiosity.add_curiosity("mom's surgery", "health", "My mom is having surgery next week", 0.6)
    print()

    # Test 2: Get high urgency topics
    print("2. High urgency topics:")
    high = curiosity.get_high_urgency_topics(threshold=0.3)
    for t in high:
        print(f"   {t.topic}: {t.urgency:.0%} urgency")
    print()

    # Test 3: Format for prompt
    print("3. Prompt context:")
    context = curiosity.format_for_prompt()
    print(context)
    print()

    # Test 4: Detect triggers
    print("4. Testing trigger detection:")
    triggers = curiosity.detect_curiosity_triggers("I have a doctor appointment next Tuesday")
    print(f"   Detected {len(triggers)} trigger(s):")
    for trigger in triggers:
        print(f"   - {trigger['topic']} ({trigger['category']}): priority {trigger['priority']:.0%}")

    print("\n=== Test Complete ===")
