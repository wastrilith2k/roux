"""
Message Validator Agent -- post-generation anti-confabulation gate.

WHAT: After the LLM generates a response but before it is sent to the user,
      this agent extracts factual claims (ClaimExtractor), verifies each one
      against entity profiles and corrections (ClaimVerifier), and produces a
      ValidationReport. High-severity contradictions can trigger regeneration.

WHY:  Even with pre-generation memory injection (MemoryValidationAgent), the
      LLM sometimes still invents facts. This is the last line of defense --
      it catches contradictions like "you work at Google" when the profile
      says otherwise, and can either flag or auto-regenerate.

HOW:  `validate()` calls extract_claims() then verify_claims(), collecting
      all contradictions. If any claim has severity >= HIGH, the report
      recommends regeneration. The message handler checks the report and
      either sends the response, regenerates, or patches the offending text.

Pipeline position: LLM response -> [this] -> send to user
Counterpart: MemoryValidationAgent (pre-generation)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .claim_extractor import ClaimExtractor, Claim, get_claim_extractor, extract_claims
from .claim_verifier import ClaimVerifier, ClaimVerificationReport, get_claim_verifier, verify_claims

logger = logging.getLogger(__name__)


@dataclass
class MessageValidationResult:
    """Result of validating a message."""
    original_response: str
    validated_response: str
    was_modified: bool = False
    claims_found: int = 0
    contradictions_found: int = 0
    contradictions: List[dict] = field(default_factory=list)
    should_regenerate: bool = False
    regeneration_hints: List[str] = field(default_factory=list)


class MessageValidatorAgent:
    """
    Validates the companion's responses before sending.

    This agent:
    1. Extracts factual claims from the companion's response
    2. Verifies claims against entity profiles and corrections
    3. Reports contradictions (for logging/debugging)
    4. Optionally suggests regeneration for high-severity issues

    Note: Currently doesn't auto-modify responses to avoid
    breaking natural conversation flow. Instead, it logs issues
    and can request regeneration for severe contradictions.
    """

    def __init__(self):
        self.extractor = get_claim_extractor()
        self.verifier = get_claim_verifier()

    def validate(
        self,
        response: str,
        user_message: str = "",
        max_regenerations: int = 1
    ) -> MessageValidationResult:
        """
        Validate the companion's response for factual accuracy.

        Args:
            response: The companion's generated response
            user_message: The user's message (for context)
            max_regenerations: How many regeneration attempts allowed

        Returns:
            MessageValidationResult with validation details
        """
        # Skip validation for very short responses (pure emotes, etc.)
        if len(response) < 20:
            return MessageValidationResult(
                original_response=response,
                validated_response=response
            )

        # Step 1: Extract claims
        claims = self.extractor.extract(response)

        if not claims:
            logger.debug("No verifiable claims found in response")
            return MessageValidationResult(
                original_response=response,
                validated_response=response
            )

        logger.info(f"Extracted {len(claims)} claims from response")

        # Step 2: Verify claims
        report = self.verifier.verify(claims)

        # Step 3: Build result
        contradictions_data = []
        regeneration_hints = []

        for c in report.contradictions:
            contradiction_info = {
                'claim': c.claim.claim_text,
                'subject': c.claim.subject,
                'severity': c.claim.severity,
                'source': c.contradiction_source,
                'correct_info': c.correct_info
            }
            contradictions_data.append(contradiction_info)

            # Build regeneration hint
            if c.claim.severity == 'high':
                regeneration_hints.append(
                    f"DO NOT say: {c.claim.claim_text}. "
                    f"CORRECT: {c.correct_info}"
                )

        # Log contradictions
        if report.contradictions:
            logger.warning(
                f"Found {len(report.contradictions)} contradictions in response"
            )
            for c in report.contradictions:
                logger.warning(
                    f"  [{c.claim.severity}] {c.claim.claim_text[:50]}... "
                    f"→ {c.correct_info[:50]}..."
                )

        return MessageValidationResult(
            original_response=response,
            validated_response=response,  # Not modifying for now
            was_modified=False,
            claims_found=report.claims_checked,
            contradictions_found=len(report.contradictions),
            contradictions=contradictions_data,
            should_regenerate=report.should_regenerate,
            regeneration_hints=regeneration_hints
        )

    def format_regeneration_instruction(
        self,
        result: MessageValidationResult
    ) -> str:
        """
        Format hints for response regeneration.

        If regeneration is needed, this provides instructions to
        include in the prompt for the retry.

        Args:
            result: Previous validation result

        Returns:
            Instruction string to add to prompt
        """
        if not result.should_regenerate or not result.regeneration_hints:
            return ""

        lines = [
            "",
            "[CRITICAL CORRECTIONS - Your previous response contained errors]",
            ""
        ]

        for hint in result.regeneration_hints:
            lines.append(f"  - {hint}")

        lines.append("")
        lines.append("Please regenerate your response avoiding these errors.")
        lines.append("")

        return "\n".join(lines)


# Singleton instance
_agent: Optional[MessageValidatorAgent] = None


def get_message_validator_agent() -> MessageValidatorAgent:
    """Get singleton MessageValidatorAgent instance."""
    global _agent
    if _agent is None:
        _agent = MessageValidatorAgent()
    return _agent


def validate_message(
    response: str,
    user_message: str = ""
) -> MessageValidationResult:
    """
    Convenience function to validate a response.

    Args:
        response: the companion's response to validate
        user_message: User's message for context

    Returns:
        MessageValidationResult
    """
    agent = get_message_validator_agent()
    return agent.validate(response, user_message)
