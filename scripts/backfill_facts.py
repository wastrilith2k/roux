#!/usr/bin/env python3
"""
Re-extract facts from existing conversation history.

Clears the current facts tables and re-runs extraction on every conversation
pair using the correct per-companion persona config. Fixes the subject naming
bug (Kai/Companion mixed) by ensuring each extraction uses the right
companion_id derived from user_email.

Usage:
    python scripts/backfill_facts.py [--companions kai mira] [--dry-run]
"""
import sys
import os
import argparse
import time

sys.path.insert(0, '/app')

COMPANIONS = ['kai', 'mira']


def get_db_conn():
    import psycopg2
    return psycopg2.connect(
        host=os.environ.get('POSTGRES_HOST', 'postgres'),
        port=os.environ.get('POSTGRES_PORT', '5432'),
        dbname=os.environ.get('POSTGRES_DB', 'companion_dev'),
        user=os.environ.get('POSTGRES_USER', 'companion'),
        password=os.environ.get('POSTGRES_PASSWORD', 'companion_dev_password')
    )


def clear_facts(cid: str, dry_run: bool):
    email = f"{cid}@companion.local"
    schema = f"user_{cid}"
    conn = get_db_conn()
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {schema}.facts WHERE user_email = %s", (email,))
        count = cur.fetchone()[0]
        print(f"  [{cid}] {count} existing facts found")
        if not dry_run:
            cur.execute(f"DELETE FROM {schema}.facts WHERE user_email = %s", (email,))
            conn.commit()
            print(f"  [{cid}] Cleared {count} facts")
        else:
            print(f"  [{cid}] DRY RUN: would delete {count} facts")
    conn.close()


def get_conversation_pairs(cid: str):
    """
    Return list of (user_msg_id, user_msg_text, companion_msg_text)
    by pairing consecutive user→companion turns within each conversation.
    """
    from src.config.persona_config import get_persona_config
    _pc = get_persona_config(companion_id=cid)
    user_name = _pc.primary_user_name.lower()
    companion_name = _pc.companion_short_name.lower()

    email = f"{cid}@companion.local"
    schema = f"user_{cid}"

    conn = get_db_conn()
    pairs = []
    with conn.cursor() as cur:
        # Get all conversations that have both user and companion messages
        cur.execute(f"""
            SELECT DISTINCT conversation_id FROM {schema}.messages
            WHERE email = %s AND conversation_id IS NOT NULL
            ORDER BY conversation_id
        """, (email,))
        conv_ids = [r[0] for r in cur.fetchall()]

        for conv_id in conv_ids:
            cur.execute(f"""
                SELECT id, sender_name, message_text
                FROM {schema}.messages
                WHERE email = %s AND conversation_id = %s
                ORDER BY id ASC
            """, (email, conv_id))
            rows = cur.fetchall()

            # Pair each user message with the immediately following companion message
            last_user_msg = None
            last_user_id = None
            for msg_id, sender, text in rows:
                sender_lower = sender.lower().strip()
                if sender_lower == user_name:
                    last_user_msg = text
                    last_user_id = msg_id
                elif sender_lower == companion_name and last_user_msg is not None:
                    pairs.append((last_user_id, last_user_msg, text))
                    last_user_msg = None
                    last_user_id = None

    conn.close()
    return pairs


def run_extraction(cid: str, pairs: list, dry_run: bool):
    from src.tasks.fact_extraction_task import extract_facts_via_llm, validate_facts, store_facts
    from src.tasks.fact_extraction_task import _companion_id_from_email

    email = f"{cid}@companion.local"
    total_extracted = 0
    total_stored = 0

    for i, (msg_id, user_msg, companion_msg) in enumerate(pairs):
        print(f"  [{cid}] Pair {i+1}/{len(pairs)} (msg_id={msg_id})...", end=' ', flush=True)
        try:
            facts = extract_facts_via_llm(user_msg, companion_msg, companion_id=cid)
            if not facts:
                print("no facts")
                continue

            validated = validate_facts(facts, companion_id=cid)
            total_extracted += len(validated)

            if dry_run:
                for f in validated:
                    print(f"\n    [{f.get('subject')}] {f.get('fact', '')[:70]}")
                print(f"  → {len(validated)} facts (dry run)")
            else:
                stored = store_facts(validated, email, msg_id)
                total_stored += stored
                print(f"{len(validated)} extracted, {stored} stored")

            # Small delay to avoid hammering the LLM API
            time.sleep(0.5)

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    return total_extracted, total_stored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--companions', nargs='+', default=COMPANIONS)
    parser.add_argument('--dry-run', action='store_true',
                        help='Extract and print facts without writing to DB')
    args = parser.parse_args()

    for cid in args.companions:
        print(f"\n{'='*50}")
        print(f"Processing: {cid}")
        print(f"{'='*50}")

        # Step 1: clear existing facts
        print("\nStep 1: Clearing existing facts...")
        clear_facts(cid, args.dry_run)

        # Step 2: load conversation pairs
        print("\nStep 2: Loading conversation pairs...")
        pairs = get_conversation_pairs(cid)
        print(f"  [{cid}] Found {len(pairs)} user→companion pairs to process")

        if not pairs:
            print(f"  [{cid}] No pairs found, skipping")
            continue

        # Step 3: re-extract
        print(f"\nStep 3: Extracting facts{' (DRY RUN)' if args.dry_run else ''}...")
        extracted, stored = run_extraction(cid, pairs, args.dry_run)

        print(f"\n  [{cid}] Done: {extracted} facts extracted, {stored} stored")

    print("\nBackfill complete.")


if __name__ == '__main__':
    main()
