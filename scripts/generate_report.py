#!/usr/bin/env python3
"""
Report Generator — Companion Framework

WHAT: Generates a self-contained HTML report summarizing simulation results,
      and optionally exports raw data as JSON.
WHY:  After running simulate_relationship.py for several weeks, you need a way to
      review what happened: how many messages were exchanged, what opinions formed,
      what curiosity threads emerged, what episodes were learned, and what reflections
      were produced. This report provides that overview in a single HTML file that
      can be opened in any browser.
HOW:  For each companion:
      1. Queries PostgreSQL for messages, opinions, curiosities, episodes,
         reflections, and fact counts
      2. Renders them into a dark-themed HTML report with stats cards,
         lists, and timeline views
      3. Optionally exports the raw data as JSON for custom visualization

Usage:
    python scripts/generate_report.py --output report.html
    python scripts/generate_report.py --output report.html --json data.json
    python scripts/generate_report.py --companions kai mira
"""

import argparse
import html as html_mod
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PST = ZoneInfo('America/Los_Angeles')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_project_root() -> Path:
    candidates = [Path(__file__).parent.parent, Path('/app'), Path.cwd()]
    for p in candidates:
        if (p / 'data').exists():
            return p
    raise RuntimeError("Cannot find project root")


def get_db():
    from src.database.db import get_db as _get_db
    return _get_db()


# ---------------------------------------------------------------------------
# Data collection — queries PostgreSQL for all reportable data
# ---------------------------------------------------------------------------

def collect_companion_data(companion_id: str) -> dict:
    """Collect all simulation data for a companion from PostgreSQL.

    Each query is wrapped in a try/except so missing tables don't crash
    the report — the simulation may not have run long enough to populate
    all tables.
    """
    db = get_db()
    data = {'id': companion_id, 'messages': [], 'opinions': [], 'curiosities': [],
            'episodes': [], 'reflections': [], 'facts_count': 0}

    # Messages
    try:
        result = db.execute(
            "SELECT role, content, created_at FROM messages WHERE companion_id = %s ORDER BY created_at",
            (companion_id,)
        )
        data['messages'] = result.fetchall()
    except Exception:
        pass

    # Opinions
    try:
        result = db.execute(
            "SELECT topic, opinion_text, confidence, created_at, updated_at FROM opinions WHERE companion_id = %s ORDER BY updated_at DESC",
            (companion_id,)
        )
        data['opinions'] = result.fetchall()
    except Exception:
        pass

    # Curiosity threads
    try:
        result = db.execute(
            "SELECT topic, urgency, resolved, created_at FROM curiosity_threads WHERE companion_id = %s ORDER BY urgency DESC",
            (companion_id,)
        )
        data['curiosities'] = result.fetchall()
    except Exception:
        pass

    # Episodes
    try:
        result = db.execute(
            "SELECT summary, learning, created_at FROM episode_learnings WHERE companion_id = %s ORDER BY created_at",
            (companion_id,)
        )
        data['episodes'] = result.fetchall()
    except Exception:
        pass

    # Reflections
    try:
        result = db.execute(
            "SELECT reflection_type, content, created_at FROM reflections WHERE companion_id = %s ORDER BY created_at",
            (companion_id,)
        )
        data['reflections'] = result.fetchall()
    except Exception:
        pass

    # Facts count (from knowledge graph or fact store)
    try:
        result = db.execute(
            "SELECT COUNT(*) as cnt FROM user_facts WHERE companion_id = %s",
            (companion_id,)
        )
        row = result.fetchone()
        data['facts_count'] = row['cnt'] if row else 0
    except Exception:
        pass

    return data


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

def generate_html_report(companions_data: list, output_path: str):
    """Generate a self-contained HTML report (no external CSS/JS dependencies)."""

    companion_sections = ""
    for comp in companions_data:
        cid = comp['id']
        msg_count = len(comp['messages'])
        opinion_count = len(comp['opinions'])
        curiosity_count = len(comp['curiosities'])
        active_curiosities = sum(1 for c in comp['curiosities'] if not c.get('resolved', True))
        episode_count = len(comp['episodes'])
        reflection_count = len(comp['reflections'])

        # Opinion list
        opinion_html = ""
        for op in comp['opinions'][:15]:
            topic = html_mod.escape(str(op.get('topic', '?')))
            text = html_mod.escape(str(op.get('opinion_text', ''))[:120])
            conf = html_mod.escape(str(op.get('confidence', '?')))
            opinion_html += f'<li><strong>{topic}</strong> (confidence: {conf}): {text}</li>\n'

        # Curiosity list
        curiosity_html = ""
        for cur in comp['curiosities'][:10]:
            topic = html_mod.escape(str(cur.get('topic', '?')))
            urgency = html_mod.escape(str(cur.get('urgency', '?')))
            status = "resolved" if cur.get('resolved') else "active"
            curiosity_html += f'<li class="{status}"><strong>{topic}</strong> (urgency: {urgency}) [{status}]</li>\n'

        # Episode timeline
        episode_html = ""
        for ep in comp['episodes'][:20]:
            summary = html_mod.escape(str(ep.get('summary', ''))[:150])
            learning = html_mod.escape(str(ep.get('learning', ''))[:150])
            date = ep.get('created_at', '')
            if hasattr(date, 'strftime'):
                date = date.strftime('%Y-%m-%d')
            else:
                date = html_mod.escape(str(date))
            episode_html += f'<div class="episode"><span class="date">{date}</span> <strong>{summary}</strong><br><em>{learning}</em></div>\n'

        # Reflections
        reflection_html = ""
        for ref in comp['reflections'][:10]:
            rtype = html_mod.escape(str(ref.get('reflection_type', '?')))
            content = html_mod.escape(str(ref.get('content', ''))[:300])
            date = ref.get('created_at', '')
            if hasattr(date, 'strftime'):
                date = date.strftime('%Y-%m-%d')
            else:
                date = html_mod.escape(str(date))
            reflection_html += f'<div class="reflection"><span class="badge">{rtype}</span> <span class="date">{date}</span><p>{content}</p></div>\n'

        companion_sections += f"""
        <section class="companion" id="{cid}">
            <h2>{cid.title()}</h2>
            <div class="stats">
                <div class="stat"><span class="number">{msg_count}</span><span class="label">Messages</span></div>
                <div class="stat"><span class="number">{comp['facts_count']}</span><span class="label">Facts Learned</span></div>
                <div class="stat"><span class="number">{opinion_count}</span><span class="label">Opinions</span></div>
                <div class="stat"><span class="number">{active_curiosities}/{curiosity_count}</span><span class="label">Active Curiosities</span></div>
                <div class="stat"><span class="number">{episode_count}</span><span class="label">Episodes</span></div>
                <div class="stat"><span class="number">{reflection_count}</span><span class="label">Reflections</span></div>
            </div>

            <h3>Opinions Formed</h3>
            <ul class="opinions">{opinion_html if opinion_html else '<li>No opinions yet</li>'}</ul>

            <h3>Curiosity Threads</h3>
            <ul class="curiosities">{curiosity_html if curiosity_html else '<li>No curiosities yet</li>'}</ul>

            <h3>Episode Timeline</h3>
            <div class="episodes">{episode_html if episode_html else '<p>No episodes yet</p>'}</div>

            <h3>Reflections</h3>
            <div class="reflections">{reflection_html if reflection_html else '<p>No reflections yet</p>'}</div>
        </section>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Companion Framework — Simulation Report</title>
    <style>
        :root {{
            --bg: #0d1117;
            --surface: #161b22;
            --border: #30363d;
            --text: #c9d1d9;
            --text-muted: #8b949e;
            --accent: #58a6ff;
            --green: #3fb950;
            --orange: #d29922;
            --purple: #bc8cff;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
        }}
        h1 {{ color: var(--accent); margin-bottom: 0.5rem; font-size: 2rem; }}
        h2 {{ color: var(--purple); margin: 2rem 0 1rem; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }}
        h3 {{ color: var(--text-muted); margin: 1.5rem 0 0.5rem; font-size: 1rem; text-transform: uppercase; letter-spacing: 0.05em; }}
        .subtitle {{ color: var(--text-muted); margin-bottom: 2rem; }}
        .companion {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1.5rem; margin-bottom: 2rem; }}
        .stats {{ display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }}
        .stat {{ background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 0.75rem 1rem; text-align: center; min-width: 100px; }}
        .stat .number {{ display: block; font-size: 1.5rem; font-weight: bold; color: var(--accent); }}
        .stat .label {{ display: block; font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; }}
        ul {{ list-style: none; padding: 0; }}
        li {{ padding: 0.5rem 0; border-bottom: 1px solid var(--border); }}
        li:last-child {{ border-bottom: none; }}
        li strong {{ color: var(--green); }}
        li.resolved {{ opacity: 0.5; }}
        .episode {{ padding: 0.75rem; border-left: 3px solid var(--accent); margin-bottom: 0.5rem; background: var(--bg); }}
        .episode .date {{ color: var(--text-muted); font-size: 0.8rem; }}
        .reflection {{ padding: 0.75rem; margin-bottom: 0.5rem; background: var(--bg); border-radius: 4px; }}
        .reflection .badge {{ background: var(--purple); color: var(--bg); padding: 0.1rem 0.5rem; border-radius: 3px; font-size: 0.75rem; font-weight: bold; }}
        .reflection .date {{ color: var(--text-muted); font-size: 0.8rem; margin-left: 0.5rem; }}
        .reflection p {{ margin-top: 0.5rem; }}
        .footer {{ margin-top: 3rem; color: var(--text-muted); font-size: 0.8rem; text-align: center; }}
    </style>
</head>
<body>
    <h1>Companion Framework</h1>
    <p class="subtitle">Simulation Report &mdash; Generated {datetime.now(PST).strftime('%Y-%m-%d %H:%M PST')}</p>

    {companion_sections}

    <div class="footer">
        <p>Generated by Companion Framework &mdash; Autonomous Conversational Agent Architecture</p>
        <p>25+ independently toggleable subsystems working in concert</p>
    </div>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)

    logger.info(f"Report generated: {output_path}")


def generate_json_export(companions_data: list, output_path: str):
    """Export raw data as JSON for custom visualization or further analysis."""

    def serialize(obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)

    with open(output_path, 'w') as f:
        json.dump(companions_data, f, default=serialize, indent=2)

    logger.info(f"JSON export: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate simulation report')
    parser.add_argument('--output', type=str, default='report.html', help='Output file path')
    parser.add_argument('--json', type=str, help='Also export raw data as JSON')
    parser.add_argument('--companions', nargs='+', required=True, help='Companion IDs (e.g. --companions kai mira)')
    args = parser.parse_args()

    project_root = find_project_root()
    os.chdir(project_root)
    sys.path.insert(0, str(project_root))

    # Collect data
    companions_data = []
    for cid in args.companions:
        logger.info(f"Collecting data for {cid}...")
        data = collect_companion_data(cid)
        companions_data.append(data)

    # Generate HTML report
    generate_html_report(companions_data, args.output)

    # Optional JSON export
    if args.json:
        generate_json_export(companions_data, args.json)

    print(f"\nReport ready: {args.output}")
    if args.json:
        print(f"JSON export: {args.json}")


if __name__ == '__main__':
    main()
