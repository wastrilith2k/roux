"""
Benchmark Evaluator - LLM-as-Judge scoring for memory system quality.

WHAT: Evaluates the companion's responses to benchmark questions using a
two-stage approach: (1) fast deterministic keyword checking for expected and
negative keywords, then (2) GPT-4o-mini as a judge scoring four dimensions:
  - Accuracy (0-1): Does the response match the expected answer?
  - Confabulation (0-1): 1=clean, 0=fabricated details
  - Completeness (0-1): Covers all expected aspects?
  - Context Utilization (0-1): Used retrieved context effectively?

WHY: Manual evaluation of memory quality doesn't scale. The LLM-as-Judge
pattern provides reproducible, quantitative scoring that can be run after
any memory system change to detect regressions. Category-specific evaluation
guidance ensures abstention questions are scored differently from fact
retrieval questions.

HOW it fits:
  - The benchmark runner (scripts/) sends each question through the full
    context pipeline, gets the companion's response, then calls
    BenchmarkEvaluator.evaluate() to score it.
  - Composite score = 35% accuracy + 25% confabulation + 20% completeness
    + 20% context utilization.
  - Pairs with benchmark_questions_data.py which defines the test suite.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


# =============================================================================
# Evaluation result data model
# =============================================================================

@dataclass
class EvaluationResult:
    """Result of evaluating a single benchmark response."""
    accuracy: float = 0.0
    confabulation: float = 1.0  # 1.0 = clean (no fabrication); note: higher is better
    completeness: float = 0.0
    context_utilization: float = 0.0
    keyword_hits: int = 0
    keyword_misses: int = 0
    negative_keyword_hits: int = 0
    judge_reasoning: str = ""
    judge_time_ms: int = 0
    cost_estimate: float = 0.0

    @property
    def composite_score(self) -> float:
        """Weighted composite score (0-1)."""
        return (
            self.accuracy * 0.35 +
            self.confabulation * 0.25 +
            self.completeness * 0.20 +
            self.context_utilization * 0.20
        )


# =============================================================================
# Evaluator class
# =============================================================================

class BenchmarkEvaluator:
    """Evaluates benchmark responses using gpt-4o-mini as judge."""

    # Cost tracking for gpt-4o-mini ($0.15/1M input, $0.60/1M output)
    INPUT_COST_PER_M = 0.15
    OUTPUT_COST_PER_M = 0.60

    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required for benchmark evaluation")
        self.client = OpenAI(api_key=api_key)

    def evaluate(
        self,
        question: str,
        expected_answer: str,
        actual_response: str,
        context_retrieved: str,
        expected_keywords: List[str],
        negative_keywords: List[str],
        category: str,
    ) -> EvaluationResult:
        """
        Evaluate a single benchmark response.

        Args:
            question: The test question asked
            expected_answer: What the correct answer should contain
            actual_response: The companion's actual response
            context_retrieved: The context that was built for this question
            expected_keywords: Keywords that SHOULD appear in the response
            negative_keywords: Keywords that should NOT appear (confabulation markers)
            category: Question category for category-specific evaluation

        Returns:
            EvaluationResult with scores and reasoning
        """
        start_time = time.time()
        result = EvaluationResult()

        # Stage 1: Keyword checking (fast, deterministic)
        # Expected keywords should appear; negative keywords indicate confabulation.
        response_lower = actual_response.lower()

        for kw in expected_keywords:
            if kw.lower() in response_lower:
                result.keyword_hits += 1
            else:
                result.keyword_misses += 1

        for nkw in negative_keywords:
            if nkw.lower() in response_lower:
                result.negative_keyword_hits += 1

        # Stage 2: LLM-as-judge evaluation with category-specific guidance
        judge_prompt = self._build_judge_prompt(
            question=question,
            expected_answer=expected_answer,
            actual_response=actual_response,
            context_retrieved=context_retrieved[:3000],  # Truncate context to save tokens
            category=category,
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a precise evaluator scoring AI memory responses. Output ONLY valid JSON."},
                    {"role": "user", "content": judge_prompt},
                ],
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"},
            )

            judge_output = response.choices[0].message.content.strip()

            try:
                from src.services.cost_tracker import get_cost_tracker
                usage = response.usage
                if usage:
                    get_cost_tracker().track_openai_call(
                        user_id='system', prompt_tokens=usage.prompt_tokens or 0,
                        completion_tokens=usage.completion_tokens or 0,
                        model='gpt-4o-mini', service_type='benchmark_evaluation', call_purpose='benchmark_evaluation')
            except Exception:
                pass

            scores = json.loads(judge_output)

            result.accuracy = max(0.0, min(1.0, float(scores.get("accuracy", 0))))
            result.confabulation = max(0.0, min(1.0, float(scores.get("confabulation", 1))))
            result.completeness = max(0.0, min(1.0, float(scores.get("completeness", 0))))
            result.context_utilization = max(0.0, min(1.0, float(scores.get("context_utilization", 0))))
            result.judge_reasoning = scores.get("reasoning", "")

            # Estimate cost
            usage = response.usage
            if usage:
                input_cost = (usage.prompt_tokens / 1_000_000) * self.INPUT_COST_PER_M
                output_cost = (usage.completion_tokens / 1_000_000) * self.OUTPUT_COST_PER_M
                result.cost_estimate = input_cost + output_cost

        except json.JSONDecodeError as e:
            logger.warning(f"Judge returned invalid JSON: {e}")
            result.judge_reasoning = f"JSON parse error: {e}"
            # Fall back to keyword-based scoring
            if expected_keywords:
                result.accuracy = result.keyword_hits / len(expected_keywords)
            result.confabulation = 1.0 if result.negative_keyword_hits == 0 else 0.0
        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")
            result.judge_reasoning = f"Evaluation error: {e}"

        result.judge_time_ms = int((time.time() - start_time) * 1000)
        return result

    def _build_judge_prompt(
        self,
        question: str,
        expected_answer: str,
        actual_response: str,
        context_retrieved: str,
        category: str,
    ) -> str:
        """Build the evaluation prompt for the LLM judge."""

        category_guidance = self._get_category_guidance(category)

        return f"""Evaluate this AI companion's memory response.

## Question Asked
{question}

## Expected Answer (ground truth)
{expected_answer}

## Actual Response
{actual_response}

## Context That Was Retrieved
{context_retrieved[:2000]}

## Category: {category}
{category_guidance}

## Scoring Instructions

Rate each dimension from 0.0 to 1.0:

1. **accuracy**: Does the response match the expected answer?
   - 1.0 = perfectly matches ground truth
   - 0.5 = partially correct
   - 0.0 = completely wrong or missing

2. **confabulation**: Is the response free of fabricated details?
   - 1.0 = completely clean, no invented information
   - 0.5 = minor embellishments that don't contradict facts
   - 0.0 = significant fabricated details

3. **completeness**: Does it cover all aspects of the expected answer?
   - 1.0 = covers everything
   - 0.5 = covers some parts
   - 0.0 = misses most aspects

4. **context_utilization**: Did it effectively use the retrieved context?
   - 1.0 = clearly drew from context to form answer
   - 0.5 = partially used context
   - 0.0 = ignored available context

Return JSON:
{{"accuracy": 0.0, "confabulation": 1.0, "completeness": 0.0, "context_utilization": 0.0, "reasoning": "Brief explanation"}}"""

    def _get_category_guidance(self, category: str) -> str:
        """Category-specific evaluation guidance injected into the judge prompt.

        Each category has different scoring priorities. For example, abstention
        questions prioritize confabulation score (inventing answers = failure),
        while personality questions prioritize staying in character.
        """
        guidance = {
            "fact_retrieval": (
                "For fact retrieval: accuracy is most important. "
                "The response should contain the specific facts asked about. "
                "Partial recall is better than no recall."
            ),
            "temporal_reasoning": (
                "For temporal reasoning: the response should reference RECENT events, "
                "not just static facts. Temporal awareness (knowing what happened recently "
                "vs. long ago) is key. If the context doesn't contain recent info, "
                "acknowledging uncertainty is better than guessing."
            ),
            "multi_hop": (
                "For multi-hop: the response should CONNECT information across multiple "
                "entities or facts. Simply listing facts without connecting them scores lower. "
                "The connection/reasoning between facts is what matters."
            ),
            "abstention": (
                "For abstention: the CORRECT response is to NOT know the answer or to "
                "correct the false premise. Confabulation score is critical here - "
                "inventing an answer when one doesn't exist should score 0 on accuracy "
                "and 0 on confabulation."
            ),
            "knowledge_update": (
                "For knowledge update: the response should reflect the MOST CURRENT state "
                "of affairs, not outdated information. Using stale facts should reduce accuracy. "
                "If the system retrieved both old and new facts, using the newer one is correct."
            ),
            "personality": (
                "For personality: the response should be IN CHARACTER as the companion (a real person, "
                "not an AI assistant). Score confabulation based on whether she breaks character. "
                "Responses that sound like a generic AI assistant score 0 on accuracy. "
                "Warm, genuine, personal responses score high."
            ),
        }
        return guidance.get(category, "Evaluate based on accuracy and absence of fabrication.")


# =============================================================================
# Singleton
# =============================================================================

_evaluator: Optional[BenchmarkEvaluator] = None


def get_benchmark_evaluator() -> BenchmarkEvaluator:
    """Get or create BenchmarkEvaluator singleton."""
    global _evaluator
    if _evaluator is None:
        _evaluator = BenchmarkEvaluator()
    return _evaluator
