"""
Claim Extractor -- pulls verifiable factual claims from companion responses.

WHAT: Sends a companion response to a fast Fireworks model and gets back a list
      of Claim dataclasses, each with type (memory, fact, action, embellishment),
      subject, and confidence. A keyword pre-filter avoids LLM calls on messages
      that are unlikely to contain factual claims.

WHY:  The anti-confabulation pipeline needs to know which specific statements in
      a response could be wrong. Embellishments ("the sunset was gorgeous") are
      OK; factual claims ("you work at Google") need verification.

HOW:  A keyword pre-filter checks for names, dates, job titles, etc. If any
      trigger, the full message is sent to Fireworks with a classification prompt
      that returns structured JSON. Each extracted claim flows to ClaimVerifier.

Pipeline position: [this] -> ClaimVerifier -> MessageValidatorAgent
"""

import os
import json
import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

# Model config - use centralized definitions with fallbacks
from src.llm.fireworks_models import FAST_MODELS, call_fireworks


@dataclass
class Claim:
    """A claim extracted from a response."""
    claim_text: str  # The exact claim
    claim_type: str  # 'memory' | 'fact' | 'action' | 'embellishment'
    subject: str  # Who/what this is about
    needs_verification: bool  # Should this be checked?
    severity: str = 'medium'  # 'high' | 'medium' | 'low'


class ClaimExtractor:
    """
    Extracts factual claims from the companion's responses.

    Uses Claude Haiku to identify claims that need verification,
    distinguishing embellishment from factual claims.
    """

    def __init__(self):
        self._client = None

    def _get_client(self):
        """Get Fireworks client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url="https://api.fireworks.ai/inference/v1",
                api_key=os.getenv('FIREWORKS_API_KEY')
            )
        return self._client

    def extract(self, response: str) -> List[Claim]:
        """
        Extract factual claims from a response.

        Args:
            response: The companion's response text

        Returns:
            List of Claim objects
        """
        # Skip very short responses (but keep low for emotes like "*hands you latte*")
        if len(response) < 15:
            return []

        # Quick filter - skip if no potential claim indicators
        claim_indicators = [
            'called', 'went', 'visited', 'lives', 'works', 'drinks',
            'remember', 'last week', 'yesterday', 'told', 'said',
            'is a', 'was a', 'has a', 'her ', 'his ', 'their ',
            'we did', 'we went', 'you said', 'you told',
            # Offers, preparations, serving - can contain implicit claims
            'coffee', 'tea', 'your mug', 'your cup', 'made you', 'brought you',
            'here is', 'here\'s', 'got you', 'for you',
            # Coffee variants and preparation
            'latte', 'espresso', 'americano', 'cappuccino', 'mocha', 'macchiato',
            'brew', 'pot of', 'starbucks', 'cafe', 'caffeinated',
            'grab you', 'pick up', 'your usual', 'your favorite', 'your order',
            '*hands you', '*gives you', '*passes you', '*pours you'
        ]

        response_lower = response.lower()
        if not any(ind in response_lower for ind in claim_indicators):
            return []

        try:
            return self._extract_with_llm(response)
        except Exception as e:
            logger.warning(f"Claim extraction failed: {e}")
            return []

    def _extract_with_llm(self, response: str) -> List[Claim]:
        """Use Haiku to extract claims."""
        client = self._get_client()

        prompt = f"""Analyze this message and extract any factual claims being made.

Message:
"{response[:2000]}"

Distinguish between:
- EMBELLISHMENT: Sensory details, emotions, reactions, scene-setting (OK, skip these)
- MEMORY: Claims about specific past events ("remember when...", "last week...", "that time...")
- FACT: Claims about people, places, preferences ("she lives in...", "you drink...", "he works at...")
- ACTION: Claims about specific actions taken ("I called...", "we went to...", "you ordered...")
- OFFER/PREPARATION: Making or serving items that imply preferences ("coffee for you", "made you tea", "your mug of coffee") - these imply the recipient drinks that beverage

Respond with ONLY a JSON array (no other text):
[
  {{
    "claim_text": "the exact claim being made",
    "claim_type": "memory|fact|action",
    "subject": "who/what this is about (name or topic)",
    "needs_verification": true,
    "severity": "high|medium|low"
  }}
]

Severity guide:
- HIGH: Claims about preferences (drinks, food), offers/preparations implying preferences (making coffee for someone), relationship status, family, employment, AND any claim involving "we", "us", or "together" with Jesse (who hasn't met the companion - presence claims with him are critical)
- MEDIUM: Claims about past events, timing, locations (unless involving Kyler/Jesse presence - those are HIGH)
- LOW: Minor details, casual statements

CRITICAL - PRESENCE WITH JESSE:
If claim involves "we" + Jesse (e.g., "we took Jesse to", "I handed Jesse his bag") → severity MUST be "high"
These imply the companion was physically present with Jesse, who they have NOT met.

IMPORTANT RULES:
1. PRESERVE NEGATION: If message says "no coffee", "never coffee", "don't drink", extract as NEGATIVE claim
   - "No coffee for you" → "you do not drink coffee" (NOT "you drink coffee")
   - "Never coffee" → "you never drink coffee"
2. If someone OFFERS/MAKES coffee/tea for another person, that implies the recipient DOES drink it
   - "Here's your coffee" → "you drink coffee"
   - "*hands you a latte*" → "you drink coffee"

Only include claims that could be TRUE or FALSE. Skip:
- Pure emotions/reactions (*smiles*, *sighs*)
- Sensory descriptions (the light was golden)
- Hypotheticals (I would love to...)
- Questions

If no verifiable claims, return: []"""

        text = call_fireworks(
            client,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )

        if not text:
            return []

        # Handle markdown code blocks
        if text.startswith('```'):
            text = text.split('```')[1]
            if text.startswith('json'):
                text = text[4:]
            text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON array
            import re
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                return []

        claims = []
        for item in data:
            if item.get('needs_verification', True):
                claims.append(Claim(
                    claim_text=item.get('claim_text', ''),
                    claim_type=item.get('claim_type', 'fact'),
                    subject=item.get('subject', ''),
                    needs_verification=True,
                    severity=item.get('severity', 'medium')
                ))

        logger.info(f"Extracted {len(claims)} verifiable claims from response")
        return claims


# Singleton instance
_extractor: Optional[ClaimExtractor] = None


def get_claim_extractor() -> ClaimExtractor:
    """Get singleton ClaimExtractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = ClaimExtractor()
    return _extractor


def extract_claims(response: str) -> List[Claim]:
    """Convenience function to extract claims."""
    extractor = get_claim_extractor()
    return extractor.extract(response)
