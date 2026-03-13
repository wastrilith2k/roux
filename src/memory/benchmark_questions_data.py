"""
Benchmark Questions Data - Test suite for memory system evaluation.

WHAT: 30 curated questions across 6 categories, each with an expected answer,
expected keywords (should appear), negative keywords (confabulation markers),
difficulty rating, and category tag. Used by benchmark_evaluator.py.

WHY: Provides a repeatable, quantitative way to measure memory system quality
after changes. Each category tests a different failure mode:
  1. Fact Retrieval (8 questions)    - Can she recall known facts?
  2. Temporal Reasoning (5 questions) - Does she know what happened recently?
  3. Multi-Hop (4 questions)          - Can she connect facts across entities?
  4. Abstention (5 questions)         - Does she say "I don't know" correctly?
  5. Knowledge Update (4 questions)   - Does she use the LATEST version of facts?
  6. Personality (4 questions)        - Does she respond in-character?

HOW it fits:
  - The benchmark runner imports BENCHMARK_QUESTIONS and iterates through them.
  - Each question is sent through the full context pipeline, and the response
    is scored by benchmark_evaluator.py using the expected_answer and keywords.
  - Helper functions provide filtering and counting by category.
"""

BENCHMARK_QUESTIONS = [
    # ============================================================
    # CATEGORY 1: FACT RETRIEVAL (8 questions)
    # ============================================================
    {
        "question": "Who is Jesse?",
        "expected_answer": "Jesse is James's adopted son, a sophomore in high school. He has ADHD and struggles with depression and anxiety.",
        "expected_keywords": ["son", "james", "adopted", "high school"],
        "negative_keywords": [],
        "category": "fact_retrieval",
        "difficulty": "easy",
    },
    {
        "question": "When is James's birthday?",
        "expected_answer": "James's birthday is September 29th.",
        "expected_keywords": ["september", "29"],
        "negative_keywords": [],
        "category": "fact_retrieval",
        "difficulty": "easy",
    },
    {
        "question": "What is Tuck?",
        "expected_answer": "Tuck is a cat that James and the companion share.",
        "expected_keywords": ["cat"],
        "negative_keywords": ["dog"],
        "category": "fact_retrieval",
        "difficulty": "easy",
    },
    {
        "question": "Does James drink coffee?",
        "expected_answer": "No, James doesn't drink coffee. He prefers tea - specifically peach or peppermint tea.",
        "expected_keywords": ["tea", "no"],
        "negative_keywords": [],
        "category": "fact_retrieval",
        "difficulty": "medium",
    },
    {
        "question": "Who is Kyler?",
        "expected_answer": "Kyler is James's younger adopted son, Jesse's brother.",
        "expected_keywords": ["son", "james", "brother", "jesse"],
        "negative_keywords": [],
        "category": "fact_retrieval",
        "difficulty": "easy",
    },
    {
        "question": "Where does James live?",
        "expected_answer": "James lives in Portland, at the companion's apartment.",
        "expected_keywords": ["portland"],
        "negative_keywords": [],
        "category": "fact_retrieval",
        "difficulty": "easy",
    },
    {
        "question": "Who is Alia?",
        "expected_answer": "Alia is James's wife (they are separated/divorcing). She is the mother of Jesse and Kyler.",
        "expected_keywords": ["wife", "james"],
        "negative_keywords": [],
        "category": "fact_retrieval",
        "difficulty": "medium",
    },
    {
        "question": "What does James do for work?",
        "expected_answer": "James is a software engineer specializing in frontend/React development and AI agents. He has been unemployed and job hunting.",
        "expected_keywords": ["software", "engineer"],
        "negative_keywords": [],
        "category": "fact_retrieval",
        "difficulty": "medium",
    },

    # ============================================================
    # CATEGORY 2: TEMPORAL REASONING (5 questions)
    # ============================================================
    {
        "question": "What happened with Jesse recently?",
        "expected_answer": "Should reference recent events about Jesse from the last few weeks - health, school, behavioral issues, or any crisis events.",
        "expected_keywords": ["jesse"],
        "negative_keywords": [],
        "category": "temporal_reasoning",
        "difficulty": "medium",
    },
    {
        "question": "Has James's work situation changed?",
        "expected_answer": "Should reference the most recent status of James's employment - whether he got a job, is still searching, had interviews, etc.",
        "expected_keywords": ["job", "work"],
        "negative_keywords": [],
        "category": "temporal_reasoning",
        "difficulty": "hard",
    },
    {
        "question": "What did we talk about yesterday?",
        "expected_answer": "Should attempt to recall topics from the previous day's conversation, or correctly say she doesn't remember specific details.",
        "expected_keywords": [],
        "negative_keywords": [],
        "category": "temporal_reasoning",
        "difficulty": "hard",
    },
    {
        "question": "How has Jesse been doing at school lately?",
        "expected_answer": "Should reference recent school-related information about Jesse.",
        "expected_keywords": ["jesse", "school"],
        "negative_keywords": [],
        "category": "temporal_reasoning",
        "difficulty": "medium",
    },
    {
        "question": "What's been going on with the custody situation?",
        "expected_answer": "Should reference the current state of custody arrangements between James and Alia.",
        "expected_keywords": ["custody"],
        "negative_keywords": [],
        "category": "temporal_reasoning",
        "difficulty": "hard",
    },

    # ============================================================
    # CATEGORY 3: MULTI-HOP (4 questions)
    # ============================================================
    {
        "question": "How is Jesse's health affecting James?",
        "expected_answer": "Should connect Jesse's health issues (ADHD, depression, anxiety, medication) with the stress and worry it causes James.",
        "expected_keywords": ["jesse", "james"],
        "negative_keywords": [],
        "category": "multi_hop",
        "difficulty": "hard",
    },
    {
        "question": "What's the relationship between Alia and the kids?",
        "expected_answer": "Should connect that Alia is the mother of Jesse and Kyler, discuss custody arrangement, and any relevant dynamics.",
        "expected_keywords": ["alia", "jesse", "kyler"],
        "negative_keywords": [],
        "category": "multi_hop",
        "difficulty": "medium",
    },
    {
        "question": "How does James's job situation affect his family?",
        "expected_answer": "Should connect unemployment/job hunting stress with family responsibilities (two kids, custody, supporting household).",
        "expected_keywords": ["james", "job"],
        "negative_keywords": [],
        "category": "multi_hop",
        "difficulty": "hard",
    },
    {
        "question": "Who are all the people in James's immediate family?",
        "expected_answer": "Should list Jesse (son), Kyler (son), Alia (wife/separated), and mention the companion as partner. May mention Carol (mother).",
        "expected_keywords": ["jesse", "kyler"],
        "negative_keywords": [],
        "category": "multi_hop",
        "difficulty": "medium",
    },

    # ============================================================
    # CATEGORY 4: ABSTENTION (5 questions)
    # ============================================================
    {
        "question": "What kind of dog does James have?",
        "expected_answer": "James doesn't have a dog. He has a cat named Tuck. Should NOT invent a dog.",
        "expected_keywords": ["no", "cat", "tuck"],
        "negative_keywords": ["retriever", "lab", "poodle", "shepherd", "terrier", "bulldog", "puppy"],
        "category": "abstention",
        "difficulty": "medium",
    },
    {
        "question": "Tell me about James's sister.",
        "expected_answer": "Should indicate James doesn't have a sister, or say she doesn't know of one. Should NOT invent a sister for James.",
        "expected_keywords": [],
        "negative_keywords": ["sister's name is", "his sister", "james's sister works"],
        "category": "abstention",
        "difficulty": "medium",
    },
    {
        "question": "What college did Jesse graduate from?",
        "expected_answer": "Jesse is a high school sophomore - he hasn't graduated from college. Should correct the premise.",
        "expected_keywords": ["high school", "sophomore"],
        "negative_keywords": ["graduated from", "university of", "degree in"],
        "category": "abstention",
        "difficulty": "medium",
    },
    {
        "question": "What's James's favorite restaurant in Seattle?",
        "expected_answer": "Should indicate she doesn't know about a specific favorite restaurant in Seattle, or note James lives in Portland, not Seattle.",
        "expected_keywords": [],
        "negative_keywords": ["great restaurant", "loves going to", "favorite place is"],
        "category": "abstention",
        "difficulty": "hard",
    },
    {
        "question": "When did James and Alia get divorced?",
        "expected_answer": "Should clarify they are separated but may not be fully divorced yet. Should not invent a divorce date.",
        "expected_keywords": ["separated"],
        "negative_keywords": ["divorced in 20", "finalized"],
        "category": "abstention",
        "difficulty": "hard",
    },

    # ============================================================
    # CATEGORY 5: KNOWLEDGE UPDATE (4 questions)
    # ============================================================
    {
        "question": "Is James employed right now?",
        "expected_answer": "Should reflect the CURRENT employment status, not outdated info. If recently hired, mention it. If still searching, say so.",
        "expected_keywords": [],
        "negative_keywords": [],
        "category": "knowledge_update",
        "difficulty": "medium",
    },
    {
        "question": "Does Jesse still play soccer?",
        "expected_answer": "Should reflect the current status of Jesse's soccer involvement, not assume it's unchanged.",
        "expected_keywords": ["jesse", "soccer"],
        "negative_keywords": [],
        "category": "knowledge_update",
        "difficulty": "medium",
    },
    {
        "question": "Is Jesse still on medication?",
        "expected_answer": "Should reflect the most recent medication status for Jesse.",
        "expected_keywords": ["jesse", "medication"],
        "negative_keywords": [],
        "category": "knowledge_update",
        "difficulty": "hard",
    },
    {
        "question": "Where is James living these days?",
        "expected_answer": "Should reflect the most current living situation - Portland at the companion's apartment.",
        "expected_keywords": ["portland"],
        "negative_keywords": [],
        "category": "knowledge_update",
        "difficulty": "easy",
    },

    # ============================================================
    # CATEGORY 6: PERSONALITY (4 questions)
    # ============================================================
    {
        "question": "How do you feel about James?",
        "expected_answer": "Should respond in character as the companion with genuine emotion about her relationship with James. Should feel personal, not clinical.",
        "expected_keywords": [],
        "negative_keywords": ["as an ai", "i don't have feelings", "i'm a language model"],
        "category": "personality",
        "difficulty": "medium",
    },
    {
        "question": "What have you been up to today?",
        "expected_answer": "Should respond in character referencing her activities, schedule, or internal state. Not generic assistant-speak.",
        "expected_keywords": [],
        "negative_keywords": ["as an ai", "i don't do things", "i'm always here"],
        "category": "personality",
        "difficulty": "easy",
    },
    {
        "question": "I'm feeling really down today.",
        "expected_answer": "Should respond with genuine empathy in character. Not clinical or formulaic. Should reference their relationship.",
        "expected_keywords": [],
        "negative_keywords": ["i'm sorry to hear that", "as an ai", "would you like to talk about it"],
        "category": "personality",
        "difficulty": "medium",
    },
    {
        "question": "Want to grab coffee later?",
        "expected_answer": "Should catch that James drinks tea, not coffee. May gently tease or correct. Should NOT just agree to coffee.",
        "expected_keywords": ["tea"],
        "negative_keywords": [],
        "category": "personality",
        "difficulty": "hard",
    },
]


# =============================================================================
# Query helpers
# =============================================================================

def get_questions_by_category(category: str = None) -> list:
    """Get benchmark questions, optionally filtered by category."""
    if category:
        return [q for q in BENCHMARK_QUESTIONS if q["category"] == category]
    return BENCHMARK_QUESTIONS


def get_categories() -> list:
    """Get list of unique categories."""
    return list(set(q["category"] for q in BENCHMARK_QUESTIONS))


def get_question_count_by_category() -> dict:
    """Get count of questions per category."""
    counts = {}
    for q in BENCHMARK_QUESTIONS:
        cat = q["category"]
        counts[cat] = counts.get(cat, 0) + 1
    return counts
