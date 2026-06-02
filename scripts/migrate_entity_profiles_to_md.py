#!/usr/bin/env python3
"""
Migrate entity profile YAML files to Markdown + front-matter format.

Usage:
    python scripts/migrate_entity_profiles_to_md.py [--dry-run] [--instances-dir instances/]

Creates .md alongside each .yaml. Does NOT delete .yaml files.
"""
import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml not installed")
    sys.exit(1)


FRONTMATTER_KEYS = {
    'name', 'age', 'occupation', 'location', 'aliases',
    'does_not_have', 'relationship_to_user', 'pronouns',
}

SECTION_MAP = {
    'personality_traits': '## Personality',
    'interests': '## Interests',
    'communication_style': '## Communication Style',
    'background': '## Background',
    'notes': '## Notes',
    'current_life': '## Current Life',
    'work': '## Work',
    'family': '## Family',
    'goals': '## Goals',
}


def _item_to_str(item) -> str:
    """Convert a list item to a readable string.

    PyYAML will parse ambiguous entries like `- Foo (bar: baz)` as a single-key
    dict {{'Foo (bar': 'baz)'}}. Reconstruct the original string in that case.
    """
    if isinstance(item, dict):
        # Single-key dict produced by an ambiguous colon in the YAML value —
        # reconstruct as "key: value".
        parts = [f"{k}: {v}" for k, v in item.items()]
        return ", ".join(parts)
    return str(item)


def yaml_to_md(yaml_path: Path) -> str:
    """Convert a YAML entity profile to Markdown + front-matter string."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    if not data:
        return ''

    fm_data = {k: v for k, v in data.items() if k in FRONTMATTER_KEYS}
    body_data = {k: v for k, v in data.items() if k not in FRONTMATTER_KEYS}

    fm_str = yaml.dump(fm_data, default_flow_style=False, allow_unicode=True).strip()

    name = data.get('name', yaml_path.stem.title())
    occupation = data.get('occupation', '')
    location = data.get('location', '')

    intro_parts = [name]
    if occupation:
        intro_parts.append(occupation)
    if location:
        intro_parts.append(f"based in {location}")

    lines = [f"# {name} — Profile", "", ", ".join(intro_parts) + ".", ""]

    for key, header in SECTION_MAP.items():
        if key in body_data:
            value = body_data.pop(key)
            lines.append(header)
            if isinstance(value, list):
                for item in value:
                    lines.append(f"- {_item_to_str(item)}")
            else:
                lines.append(str(value))
            lines.append("")

    # Remaining unmapped keys
    for key, value in body_data.items():
        header = f"## {key.replace('_', ' ').title()}"
        lines.append(header)
        if isinstance(value, list):
            for item in value:
                lines.append(f"- {_item_to_str(item)}")
        elif isinstance(value, dict):
            for k, v in value.items():
                lines.append(f"- **{k}**: {v}")
        else:
            lines.append(str(value))
        lines.append("")

    body = '\n'.join(lines).strip()
    return f"---\n{fm_str}\n---\n\n{body}\n"


def migrate(instances_dir: Path, dry_run: bool = False) -> int:
    count = 0
    for yaml_file in sorted(instances_dir.rglob("entity_profiles/*.yaml")):
        md_file = yaml_file.with_suffix('.md')
        if md_file.exists():
            print(f"  SKIP (exists): {md_file.relative_to(instances_dir.parent)}")
            continue
        content = yaml_to_md(yaml_file)
        if not content:
            print(f"  WARN (empty): {yaml_file}")
            continue
        rel = yaml_file.relative_to(instances_dir.parent)
        print(f"  {'DRY-RUN' if dry_run else 'WRITE'}: {rel.with_suffix('.md')}")
        if not dry_run:
            md_file.write_text(content, encoding='utf-8')
        count += 1
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--instances-dir', default='instances')
    args = parser.parse_args()
    instances_path = Path(args.instances_dir)
    if not instances_path.exists():
        print(f"ERROR: {instances_path} does not exist")
        sys.exit(1)
    n = migrate(instances_path, dry_run=args.dry_run)
    print(f"\n{'Would migrate' if args.dry_run else 'Migrated'} {n} profile(s).")
