"""
Entity Profile Fact Validator - Enforce YAML profiles as source of truth.

WHAT: Loads all YAML entity profiles (the hand-curated ground truth), gets all
      active facts for each profiled entity, then uses LLM to identify direct
      contradictions. Contradictory facts are archived with the reason
      'contradicted_by_entity_profile'. Triggers biography regeneration for
      any affected entities so stale bio text is refreshed.

WHEN: Daily at 3:30 AM Pacific (after biography refresh task).

WHY:  LLM extraction is imperfect. It sometimes creates facts that contradict
      known ground truth (e.g., "the companion has a daughter" when she has no
      children). YAML profiles are manually maintained and authoritative. This
      task acts as a cleanup sweep, catching extraction errors that slipped
      through the real-time validation in fact_extraction_task.

Supports dry_run mode for safe testing (report contradictions without archiving).
"""

import os
import json
import logging
from typing import Dict, List, Any

from src.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    name='tasks.entity_profile_fact_validator.validate_facts_against_profiles',
    bind=True,
    max_retries=1,
    soft_time_limit=180,
    time_limit=240
)
def validate_facts_against_profiles(self, dry_run: bool = False):
    """
    Check all active facts against entity profiles and archive contradictions.

    Args:
        dry_run: If True, only report contradictions without archiving.

    Returns:
        Dict with validation results per entity.
    """
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        import yaml
        from src.core.entity_profile_loader import EntityProfileLoader

        loader = EntityProfileLoader()

        if not loader.profiles:
            logger.warning("No entity profiles found - skipping validation")
            return {'status': 'skipped', 'reason': 'no_profiles'}

        conn = psycopg2.connect(
            host=os.environ.get('POSTGRES_HOST', 'postgres'),
            port=os.environ.get('POSTGRES_PORT', '5432'),
            dbname=os.environ.get('POSTGRES_DB', 'companion'),
            user=os.environ.get('POSTGRES_USER', 'companion'),
            password=os.environ.get('POSTGRES_PASSWORD', '')
        )

        results = {}
        total_archived = 0
        entities_affected = []

        for entity_name, profile in loader.profiles.items():
            display_name = profile.get('name', entity_name.title())
            # Also search by first name and profile key (e.g., "User" vs "User Name")
            first_name = display_name.split()[0] if display_name else entity_name.title()
            search_names = list({display_name.lower(), first_name.lower(), entity_name.lower()})

            # Get active facts for this entity (match any name variant)
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT id, subject, predicate, object
                    FROM facts
                    WHERE archived_at IS NULL
                      AND LOWER(subject) = ANY(%s)
                    ORDER BY id
                """, (search_names,))
                facts = [dict(row) for row in cursor.fetchall()]

            if not facts:
                results[entity_name] = {
                    'facts_checked': 0,
                    'contradictions': 0,
                    'archived': 0
                }
                continue

            # Use LLM to find contradictions
            contradictions = _find_contradictions(profile, facts)
            archived_count = 0

            if contradictions:
                contradiction_ids = [c['fact_id'] for c in contradictions]

                if not dry_run:
                    with conn.cursor() as cursor:
                        cursor.execute("""
                            UPDATE facts
                            SET archived_at = CURRENT_TIMESTAMP,
                                archive_reason = 'contradicted_by_entity_profile'
                            WHERE id = ANY(%s)
                              AND archived_at IS NULL
                        """, (contradiction_ids,))
                        archived_count = cursor.rowcount
                        conn.commit()
                else:
                    archived_count = 0

                total_archived += archived_count if not dry_run else len(contradictions)
                entities_affected.append(display_name)

                for c in contradictions:
                    logger.info(
                        f"{'[DRY RUN] Would archive' if dry_run else 'Archived'} "
                        f"fact #{c['fact_id']} for {display_name}: "
                        f"\"{c['fact_text']}\" - Reason: {c['reason']}"
                    )

            results[entity_name] = {
                'facts_checked': len(facts),
                'contradictions': len(contradictions),
                'archived': archived_count if not dry_run else 0,
                'details': [
                    {'id': c['fact_id'], 'text': c['fact_text'], 'reason': c['reason']}
                    for c in contradictions
                ]
            }

        conn.close()

        # Trigger biography regeneration for affected entities
        if entities_affected and not dry_run:
            _regenerate_biographies(entities_affected)

        mode = "DRY RUN" if dry_run else "EXECUTED"
        logger.info(
            f"Entity profile fact validation [{mode}]: "
            f"{len(loader.profiles)} profiles checked, "
            f"{total_archived} facts {'would be ' if dry_run else ''}archived, "
            f"{len(entities_affected)} entities affected"
        )

        return {
            'status': 'success',
            'dry_run': dry_run,
            'profiles_checked': len(loader.profiles),
            'total_archived': total_archived,
            'entities_affected': entities_affected,
            'details': results,
        }

    except Exception as e:
        logger.error(f"Entity profile fact validation failed: {e}", exc_info=True)
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=300)
        return {'status': 'error', 'error': str(e)}


def _find_contradictions(profile: Dict, facts: List[Dict]) -> List[Dict]:
    """
    Use LLM to identify facts that contradict the entity profile.

    Args:
        profile: The YAML entity profile dict.
        facts: List of active fact dicts (id, subject, predicate, object).

    Returns:
        List of dicts with fact_id, fact_text, and reason for each contradiction.
    """
    import yaml

    # Format profile as readable YAML
    profile_text = yaml.dump(profile, default_flow_style=False, allow_unicode=True)

    # Format facts as numbered list
    facts_text = "\n".join(
        f"[{f['id']}] {f['subject']}: {f.get('predicate', '')} - {f.get('object', '')}"
        for f in facts
    )

    entity_name = profile.get('name', 'Unknown')

    prompt = f"""You are a fact-checker. Compare these extracted facts against the authoritative entity profile for {entity_name}.

ENTITY PROFILE (source of truth):
```yaml
{profile_text}
```

EXTRACTED FACTS (may contain errors):
{facts_text}

Identify any facts that DIRECTLY CONTRADICT the entity profile. Only flag clear contradictions, not minor differences in wording or facts about topics the profile doesn't cover.

Examples of contradictions:
- Profile says "Kyler has met the companion" but fact says "Has not met the companion in person yet"
- Profile says gender is "male" but fact says "she/her"
- Profile says parents are ["Alex", "Jordan"] but fact says "biological child of Alex and Sam"

NOT contradictions:
- Fact mentions something the profile doesn't cover (additional info is fine)
- Minor wording differences that convey the same meaning
- Facts about events/plans (profiles cover stable traits, not transient events)

Return a JSON array of contradictions. Each element should have:
- "fact_id": the number in brackets
- "fact_text": the full fact text
- "reason": brief explanation of the contradiction

If there are NO contradictions, return an empty array: []

Return ONLY the JSON array, no other text."""

    try:
        from src.llm.provider_factory import generate_sync

        response = generate_sync(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2048,
        )

        # Parse JSON from response
        response = response.strip()
        # Handle markdown code blocks
        if response.startswith("```"):
            response = response.split("\n", 1)[1]
            response = response.rsplit("```", 1)[0]
        response = response.strip()

        contradictions = json.loads(response)

        if not isinstance(contradictions, list):
            logger.warning(f"LLM returned non-list response: {response[:200]}")
            return []

        # Validate each entry has required fields
        valid = []
        for c in contradictions:
            if isinstance(c, dict) and 'fact_id' in c:
                valid.append({
                    'fact_id': int(c['fact_id']),
                    'fact_text': str(c.get('fact_text', '')),
                    'reason': str(c.get('reason', 'Contradicts entity profile')),
                })
        return valid

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        return []
    except Exception as e:
        logger.error(f"LLM call failed during contradiction check: {e}")
        return []


def _regenerate_biographies(entity_names: List[str]):
    """Trigger biography regeneration for affected entities."""
    try:
        from src.memory.synthesized_biographies import get_biography_synthesizer
        synthesizer = get_biography_synthesizer()

        for name in entity_names:
            try:
                paragraphs = synthesizer.synthesize_for_subject(name)
                logger.info(
                    f"Regenerated {len(paragraphs)} biography paragraphs for {name}"
                )
            except Exception as e:
                logger.error(f"Failed to regenerate biography for {name}: {e}")
    except Exception as e:
        logger.error(f"Biography regeneration failed: {e}")
