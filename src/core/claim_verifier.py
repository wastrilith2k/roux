"""
Claim Verifier -- LLM-based contradiction detection for companion responses.

WHAT: Takes factual claims extracted by ClaimExtractor and checks each one
      against entity profiles (YAML) and past corrections (PostgreSQL) using
      a Fireworks LLM call. Returns a VerificationResult per claim indicating
      whether it contradicts known facts.

WHY:  The companion sometimes confabulates -- inventing details about the user's
      life, job, family, etc. This verifier catches contradictions before the
      response reaches the user, enabling the MessageValidatorAgent to flag or
      regenerate problematic responses.

HOW:  For each claim, a detailed prompt is built containing the claim text,
      relevant entity profile sections, and any stored corrections on the same
      subject. The LLM classifies the claim as VERIFIED, CONTRADICTED, or
      UNVERIFIABLE and provides the contradicting fact if applicable. Uses the
      Fireworks kimi-k2 model for quality verification with many few-shot
      examples in the prompt.

Pipeline position: ClaimExtractor -> [this] -> MessageValidatorAgent
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .claim_extractor import Claim

logger = logging.getLogger(__name__)

# Use Fireworks model for quality verification
from src.config.models import FIREWORKS_DEFAULT_MODEL as FIREWORKS_MODEL


@dataclass
class VerificationResult:
    """Result of verifying a single claim."""
    claim: Claim
    is_valid: bool
    contradiction_found: bool = False
    contradiction_source: str = ""  # 'entity_profile' | 'correction'
    correct_info: str = ""  # What the correct information is
    confidence: float = 0.5


@dataclass
class ClaimVerificationReport:
    """Full verification report for a response."""
    claims_checked: int
    valid_claims: int
    contradictions: List[VerificationResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    should_regenerate: bool = False


class ClaimVerifier:
    """
    LLM-based claim verification against known facts.

    Priority order (highest to lowest):
    1. Entity profiles (YAML) - Absolute source of truth
    2. Recent corrections - User-provided fixes (PostgreSQL)
    """

    def __init__(self):
        self._profile_loader = None
        self._correction_store = None
        self._client = None

    def _get_client(self):
        """Get Fireworks client for LLM verification."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url="https://api.fireworks.ai/inference/v1",
                api_key=os.getenv('FIREWORKS_API_KEY')
            )
        return self._client

    def _get_profile_loader(self):
        """Get entity profile loader."""
        if self._profile_loader is None:
            from src.core.entity_profile_loader import get_entity_profile_loader
            self._profile_loader = get_entity_profile_loader()
        return self._profile_loader

    def _get_correction_store(self):
        """Get correction store for PostgreSQL queries."""
        if self._correction_store is None:
            from src.core.correction_store import get_correction_store
            self._correction_store = get_correction_store()
        return self._correction_store

    def verify(self, claims: List[Claim]) -> ClaimVerificationReport:
        """
        Verify a list of claims against known facts.

        Args:
            claims: List of claims to verify

        Returns:
            ClaimVerificationReport with results
        """
        if not claims:
            return ClaimVerificationReport(
                claims_checked=0,
                valid_claims=0
            )

        results = []
        contradictions = []
        warnings = []

        for claim in claims:
            result = self._verify_claim(claim)
            results.append(result)

            if result.contradiction_found:
                contradictions.append(result)
                logger.warning(
                    f"Contradiction found: {claim.claim_text[:50]}... "
                    f"(source: {result.contradiction_source})"
                )

        # Determine if we should regenerate
        # Regenerate if high-severity contradictions found
        should_regenerate = any(
            c.claim.severity == 'high' and c.contradiction_found
            for c in results
        )

        valid_count = sum(1 for r in results if r.is_valid)

        return ClaimVerificationReport(
            claims_checked=len(claims),
            valid_claims=valid_count,
            contradictions=contradictions,
            warnings=warnings,
            should_regenerate=should_regenerate
        )

    def _verify_claim(self, claim: Claim) -> VerificationResult:
        """
        Verify a single claim against all sources.

        Checks in priority order:
        1. Entity profiles (YAML)
        2. Recent corrections
        """
        # 1. Check against entity profiles (highest priority)
        profile_result = self._check_entity_profiles(claim)
        if profile_result.contradiction_found:
            return profile_result

        # 2. Check against recent corrections
        correction_result = self._check_corrections(claim)
        if correction_result.contradiction_found:
            return correction_result

        # No contradictions found
        return VerificationResult(
            claim=claim,
            is_valid=True,
            confidence=0.7
        )

    def _check_entity_profiles(self, claim: Claim) -> VerificationResult:
        """Check claim against YAML entity profiles using LLM."""
        from src.config.persona_config import get_persona_config
        _pc = get_persona_config()
        _companion_key = _pc.companion_entity_profile
        _user_key = _pc.primary_user_entity_profile

        loader = self._get_profile_loader()
        claim_lower = claim.claim_text.lower()
        subject_lower = claim.subject.lower()

        # Determine which profiles to check
        profiles_to_check = []
        for entity_name in loader.profiles.keys():
            if entity_name in claim_lower or entity_name in subject_lower:
                profile = loader.get_profile(entity_name)
                if profile:
                    profiles_to_check.append((entity_name, profile))

        # SPECIAL CASE: Identify who the subject refers to
        # The companion is the responder, so first-person subjects refer to them
        companion_subjects = ['i', 'me', 'myself', 'the speaker', 'speaker']
        # The user is the recipient, so second-person and vague subjects refer to them
        user_subjects = ['the person', 'you', 'them', 'someone', 'recipient', 'user', 'he', 'him']

        # If subject refers to the companion, check against companion's profile (if not already)
        if subject_lower in companion_subjects:
            if not any(name == _companion_key for name, _ in profiles_to_check):
                companion_profile = loader.get_profile(_companion_key)
                if companion_profile:
                    profiles_to_check.append((_companion_key, companion_profile))
        # If subject refers to the user or is ambiguous, check user's profile
        elif subject_lower in user_subjects or not any(entity in subject_lower for entity in loader.profiles.keys()):
            if not any(name == _user_key for name, _ in profiles_to_check):
                user_profile = loader.get_profile(_user_key)
                if user_profile:
                    profiles_to_check.append((_user_key, user_profile))

        # Check each relevant profile using LLM
        for entity_name, profile in profiles_to_check:
            # Extract critical facts from profile
            facts = self._extract_critical_facts(entity_name, profile)

            # If we have facts for this entity, trust the LLM to determine relevance
            # The LLM is instructed to return false if no facts relate to the claim
            if not facts:
                logger.debug(f"No facts extracted for {entity_name}")
                continue

            # Use LLM to check for contradiction
            contradiction = self._check_contradiction_with_llm(
                claim.claim_text,
                entity_name,
                facts
            )

            if contradiction:
                return VerificationResult(
                    claim=claim,
                    is_valid=False,
                    contradiction_found=True,
                    contradiction_source='entity_profile',
                    correct_info=contradiction,
                    confidence=1.0  # Profiles are definitive
                )

        return VerificationResult(claim=claim, is_valid=True)

    def _extract_critical_facts(self, entity_name: str, profile: Dict[str, Any]) -> str:
        """
        Extract critical facts from a profile for verification.

        Focuses on facts that are commonly confabulated:
        - Drink preferences
        - Employment
        - Relationships
        - Parenting roles
        """
        facts = []
        name = profile.get('name', entity_name.title())

        # Preferences (drinks, food, etc.)
        preferences = profile.get('preferences', {})
        if preferences:
            drinks = preferences.get('drinks', {})
            if drinks:
                for drink_type, detail in drinks.items():
                    facts.append(f"{name}'s {drink_type} preference: {detail}")

        # Employment
        employment = profile.get('employment', {})
        if employment:
            if employment.get('current_employer'):
                facts.append(f"{name}'s current employer: {employment['current_employer']}")
            if employment.get('work_status'):
                facts.append(f"{name}'s work status: {employment['work_status']}")
            if employment.get('previous_employer'):
                facts.append(f"{name}'s previous employer: {employment['previous_employer']}")

        # Family relationships
        family = profile.get('family', {})
        if family:
            if family.get('spouse'):
                facts.append(f"{name}'s spouse: {family['spouse']}")
            if family.get('children'):
                for child in family['children']:
                    # Handle both string children and dict children
                    if isinstance(child, str):
                        facts.append(f"{name}'s child: {child}")
                    elif isinstance(child, dict):
                        facts.append(f"{name}'s child: {child.get('name')} ({child.get('relationship', '')})")
            # For kids' profiles - relationship with the companion (critical for confabulation prevention)
            from src.config.persona_config import get_persona_config
            _pc_local = get_persona_config()
            _companion_name = _pc_local.companion_short_name
            if family.get('relationship_with_companion'):
                facts.append(f"{name}'s relationship with {_companion_name}: {family['relationship_with_companion']}")

        # Role clarification (critical for the companion)
        role_clarification = profile.get('role_clarification', {})
        if role_clarification:
            is_not = role_clarification.get('is_not', [])
            for not_role in is_not:
                facts.append(f"{name} is NOT: {not_role}")
            is_instead = role_clarification.get('is_instead', '')
            if is_instead:
                facts.append(f"{name} IS: {is_instead}")

        # Sibling relationships
        siblings = profile.get('siblings', {})
        if siblings:
            for sib_key, sib_info in siblings.items():
                if isinstance(sib_info, dict):
                    rel_status = sib_info.get('relationship_status', '') or sib_info.get('relationship', '')
                    if rel_status:
                        facts.append(f"{name}'s relationship with {sib_key}: {rel_status}")

        # Romantic relationship
        romantic = profile.get('romantic_relationship', {})
        if romantic:
            if romantic.get('partner'):
                facts.append(f"{name}'s romantic partner: {romantic['partner']}")
            if romantic.get('status'):
                facts.append(f"{name}'s relationship status: {romantic['status']}")

        return "\n".join(facts) if facts else ""

    def _check_contradiction_with_llm(
        self,
        claim_text: str,
        entity_name: str,
        known_facts: str
    ) -> Optional[str]:
        """
        Use LLM to check if a claim contradicts known facts.

        Returns the correct information if contradiction found, None otherwise.
        """
        try:
            client = self._get_client()

            prompt = f"""You are a fact-checker. Your job is to detect CONTRADICTIONS ONLY.

CLAIM: "{claim_text}"

KNOWN FACTS ABOUT {entity_name.upper()}:
{known_facts}

=== CRITICAL: UNDERSTAND WHAT A CONTRADICTION IS ===

A CONTRADICTION means the claim ASSERTS THE OPPOSITE of a known fact.

EXAMPLES OF **NOT** CONTRADICTIONS (these AGREE or are NEUTRAL):
- Claim: "you drink tea" + Fact: "loves tea" → AGREES → contradicts: false
- Claim: "the recipient drinks tea" + Fact: "loves tea" → AGREES → contradicts: false
- Claim: "you prefer tea" + Fact: "loves tea" → AGREES → contradicts: false
- Claim: "you do not drink coffee" + Fact: "does NOT drink coffee" → AGREES → contradicts: false
- Claim: "you never drink coffee" + Fact: "does NOT drink coffee" → AGREES → contradicts: false
- Claim: "no coffee for you" + Fact: "does NOT drink coffee" → AGREES → contradicts: false
- Claim: "never coffee for you" + Fact: "does NOT drink coffee" → AGREES → contradicts: false

EXAMPLES OF **TRUE** CONTRADICTIONS (these say the OPPOSITE):
- Claim: "you drink coffee" + Fact: "does NOT drink coffee" → OPPOSITE → contradicts: true
- Claim: "here's your coffee" + Fact: "does NOT drink coffee" → implies drinking coffee → contradicts: true
- Claim: "made you a latte" + Fact: "does NOT drink coffee" → latte IS coffee → contradicts: true
- Claim: "your usual espresso" + Fact: "does NOT drink coffee" → espresso IS coffee → contradicts: true
- Claim: "we dropped Jesse off" + Fact: "Jesse has NOT met the companion" → "we" implies the companion was there → contradicts: true
- Claim: "I handed Jesse his bag" + Fact: "Jesse has NOT met the companion" → "I" = the companion physically present → contradicts: true

=== KEY RULES ===
1. ONLY use facts that are EXPLICITLY listed above - DO NOT invent facts
2. If no fact above relates to the claim topic → contradicts: false (no relevant fact to contradict)
3. "never" = "does not" = "no" - these are EQUIVALENT, not contradictions
4. If claim and fact BOTH deny something ("never drinks X" vs "does NOT drink X") → NOT a contradiction
5. Only compare SAME attributes (coffee facts vs coffee claims, tea facts vs tea claims)
6. "Drinks peppermint tea" does NOT contradict "does NOT drink coffee" - different things!
7. If claim implies person DOES consume something they DON'T consume → IS a contradiction
8. Latte, espresso, americano, cappuccino, mocha = coffee drinks
9. "Made you X" or "here's your X" implies the person consumes X
10. CRITICAL: This claim is from the companion's perspective. "We" = the companion + the user. So "we took X" or "we went with X" means the companion was present with X
11. If fact says "X has NOT met the companion" and claim implies the companion was present with X (via "we", "us", "together", "I handed X", "I gave X") → IS a contradiction

CRITICAL RULES FOR YOUR RESPONSE:
- If claim AGREES with a fact → contradicts: false
- If claim SUPPORTS or MATCHES a fact → contradicts: false
- If no fact relates to the claim → contradicts: false
- ONLY if claim says OPPOSITE of a fact → contradicts: true

/no_think
Respond with ONLY valid JSON:
{{"contradicts": true/false, "reason": "explanation", "correct_fact": "only fill if contradicts is true, otherwise empty string"}}"""

            response = client.chat.completions.create(
                model=FIREWORKS_MODEL,
                max_tokens=200,
                temperature=0,  # Deterministic for fact-checking
                messages=[{"role": "user", "content": prompt}]
            )

            text = response.choices[0].message.content.strip()

            # Handle <think> tags from reasoning models
            if '<think>' in text:
                if '</think>' in text:
                    text = text.split('</think>')[-1].strip()

            # Handle markdown code blocks
            if text.startswith('```'):
                text = text.split('```')[1]
                if text.startswith('json'):
                    text = text[4:]
                text = text.strip()

            # Parse JSON response
            try:
                result = json.loads(text)
            except json.JSONDecodeError:
                # Try to extract JSON
                import re
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if match:
                    result = json.loads(match.group())
                else:
                    logger.warning(f"Failed to parse LLM response: {text[:100]}")
                    return None

            if result.get('contradicts', False):
                correct_fact = result.get('correct_fact', '')
                reason = result.get('reason', '')
                return f"{correct_fact}" if correct_fact else reason

            return None

        except Exception as e:
            logger.warning(f"LLM verification failed: {e}")
            return None

    def _check_corrections(self, claim: Claim) -> VerificationResult:
        """Check claim against recent corrections in PostgreSQL using LLM."""
        store = self._get_correction_store()

        try:
            # Search for corrections related to claim subject
            corrections = store.get_corrections_for_subject(
                claim.subject,
                limit=5
            )

            if not corrections:
                return VerificationResult(claim=claim, is_valid=True)

            # Format corrections for LLM
            corrections_text = "\n".join([
                f"- WRONG: {c['wrong_claim']} → CORRECT: {c['correct_info']}"
                for c in corrections
            ])

            # Use LLM to check if claim repeats any wrong info
            contradiction = self._check_correction_with_llm(
                claim.claim_text,
                corrections_text
            )

            if contradiction:
                return VerificationResult(
                    claim=claim,
                    is_valid=False,
                    contradiction_found=True,
                    contradiction_source='correction',
                    correct_info=contradiction,
                    confidence=0.9
                )

        except Exception as e:
            logger.warning(f"Error checking corrections: {e}")

        return VerificationResult(claim=claim, is_valid=True)

    def _check_correction_with_llm(
        self,
        claim_text: str,
        corrections_text: str
    ) -> Optional[str]:
        """
        Use LLM to check if a claim repeats previously corrected misinformation.

        Returns the correct information if contradiction found, None otherwise.
        """
        try:
            client = self._get_client()

            prompt = f"""You are a fact-checker. Determine if the following claim repeats any previously corrected misinformation.

CLAIM: "{claim_text}"

PREVIOUS CORRECTIONS (things that were wrong and their corrections):
{corrections_text}

Does the claim repeat any of the WRONG information that was previously corrected?

/no_think
Respond with ONLY valid JSON (no markdown, no explanation, no thinking):
{{"repeats_wrong": true/false, "correct_info": "the correct information if it repeats wrong info, otherwise empty string"}}"""

            response = client.chat.completions.create(
                model=FIREWORKS_MODEL,
                max_tokens=150,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )

            text = response.choices[0].message.content.strip()

            # Handle <think> tags from reasoning models
            if '<think>' in text:
                if '</think>' in text:
                    text = text.split('</think>')[-1].strip()

            # Handle markdown code blocks
            if text.startswith('```'):
                text = text.split('```')[1]
                if text.startswith('json'):
                    text = text[4:]
                text = text.strip()

            try:
                result = json.loads(text)
            except json.JSONDecodeError:
                import re
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if match:
                    result = json.loads(match.group())
                else:
                    return None

            if result.get('repeats_wrong', False):
                return result.get('correct_info', 'See previous corrections')

            return None

        except Exception as e:
            logger.warning(f"LLM correction check failed: {e}")
            return None


# Singleton instance
_verifier: Optional[ClaimVerifier] = None


def get_claim_verifier() -> ClaimVerifier:
    """Get singleton ClaimVerifier instance."""
    global _verifier
    if _verifier is None:
        _verifier = ClaimVerifier()
    return _verifier


def verify_claims(claims: List[Claim]) -> ClaimVerificationReport:
    """Convenience function to verify claims."""
    verifier = get_claim_verifier()
    return verifier.verify(claims)
