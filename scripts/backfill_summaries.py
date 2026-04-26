#!/usr/bin/env python3
"""Back-fill daily summaries and reflections for days that have messages but no summary."""
import sys
import os
sys.path.insert(0, '/app')

from src.tasks.daily_summary_task import generate_daily_summary
from src.tasks.reflection_task import reflect_on_day

COMPANIONS = ['kai', 'mira']

# Days with messages per companion (from DB query)
DAYS = [
    '2026-01-02', '2026-01-04', '2026-01-05', '2026-01-06',
    '2026-01-08', '2026-01-09', '2026-01-11', '2026-01-12',
    '2026-01-14', '2026-01-15', '2026-01-16', '2026-01-17',
]

def _summary_exists(cid, date):
    """Check if summary already exists in DB."""
    import psycopg2
    conn = psycopg2.connect(
        host=os.environ.get('POSTGRES_HOST', 'postgres'),
        port=os.environ.get('POSTGRES_PORT', '5432'),
        dbname=os.environ.get('POSTGRES_DB', 'companion_dev'),
        user=os.environ.get('POSTGRES_USER', 'companion'),
        password=os.environ.get('POSTGRES_PASSWORD', 'companion_dev_password')
    )
    schema = f"user_{cid}"
    email = f"{cid}@companion.local"
    with conn.cursor() as cur:
        cur.execute(f"SELECT 1 FROM {schema}.daily_summaries WHERE user_email = %s AND summary_date = %s", (email, date))
        exists = cur.fetchone() is not None
    conn.close()
    return exists

def run_summary(cid, date):
    if _summary_exists(cid, date):
        print(f"[{cid}] Summary for {date} already exists, skipping")
        return True
    email = f"{cid}@companion.local"
    print(f"\n[{cid}] Generating summary for {date}...")
    result = generate_daily_summary.apply(
        kwargs={'user_email': email, 'target_date': date, 'companion_id': cid}
    )
    r = result.get(timeout=180)
    status = r.get('status') if isinstance(r, dict) else str(r)
    wc = r.get('word_count', '') if isinstance(r, dict) else ''
    print(f"  → {status} {wc}")
    return status == 'success'

def run_reflection(cid, date):
    email = f"{cid}@companion.local"
    print(f"[{cid}] Reflecting on {date}...")
    result = reflect_on_day.apply(
        kwargs={'user_email': email, 'target_date': date, 'companion_id': cid}
    )
    r = result.get(timeout=180)
    status = r.get('status') if isinstance(r, dict) else str(r)
    print(f"  → {status}")
    return status == 'success'

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--companions', nargs='+', default=COMPANIONS)
    parser.add_argument('--skip-reflection', action='store_true')
    args = parser.parse_args()

    for cid in args.companions:
        print(f"\n{'='*50}")
        print(f"Back-filling {cid}")
        print(f"{'='*50}")
        for date in DAYS:
            ok = run_summary(cid, date)
            if ok and not args.skip_reflection:
                run_reflection(cid, date)

    print("\nBack-fill complete.")
