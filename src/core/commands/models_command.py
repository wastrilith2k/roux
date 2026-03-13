"""
/models command - Shows which LLM models are used for different tasks
"""

import os
from typing import Dict, Any


def get_model_info() -> Dict[str, Any]:
    """
    Get information about all models used in the system.

    Returns dict with:
    - models: List of {task, model, provider, notes}
    - summary: Quick summary text
    """
    models = []

    # Main conversation
    llm_provider = os.environ.get("LLM_PROVIDER", "fireworks")
    if llm_provider == "fireworks":
        main_model = os.environ.get(
            "FIREWORKS_MODEL",
            "accounts/fireworks/models/kimi-k2-instruct-0905"
        )
        models.append({
            'task': 'Main Conversation',
            'model': main_model.split('/')[-1],
            'provider': 'Fireworks',
            'notes': 'Primary response generation'
        })
    else:
        main_model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        models.append({
            'task': 'Main Conversation',
            'model': main_model,
            'provider': 'Anthropic',
            'notes': 'Primary response generation'
        })

    # Fallback conversation
    models.append({
        'task': 'Fallback Conversation',
        'model': 'claude-sonnet-4-20250514',
        'provider': 'Anthropic',
        'notes': 'If primary fails'
    })

    # Heavy processing tasks (Kimi K2)
    heavy_tasks = [
        ('Fact Extraction', 'Extracts facts from conversations'),
        ('Importance Scoring', 'Rates fact importance 1-10'),
        ('Value Inference', 'Companion\'s hidden feelings'),
        ('Reach-Out Decisions', 'Decides when to message'),
        ('Claim Verification', 'Verifies memory claims'),
        ('Image Intent Detection', 'Detects image requests'),
    ]

    for task, notes in heavy_tasks:
        models.append({
            'task': task,
            'model': 'kimi-k2-instruct-0905',
            'provider': 'Fireworks',
            'notes': notes
        })

    # Light/fast tasks (Llama 8B)
    light_tasks = [
        ('Claim Extraction', 'Identifies claims to verify'),
        ('Correction Detection', 'Detects user corrections'),
        ('Memory Query Classification', 'Classifies memory queries'),
        ('Prompt Enhancement', 'Enhances image prompts'),
        ('Fact Review', 'Quick fact validation'),
    ]

    for task, notes in light_tasks:
        models.append({
            'task': task,
            'model': 'llama-v3p1-8b-instruct',
            'provider': 'Fireworks',
            'notes': notes
        })

    # Anthropic tasks
    models.append({
        'task': 'Scene Tracking',
        'model': 'claude-haiku-4-20250514',
        'provider': 'Anthropic',
        'notes': 'Roleplay state management'
    })

    # OpenAI tasks
    openai_tasks = [
        ('Tool Execution', 'Function calling for tools'),
        ('Message Condensing', 'Compresses verbose messages'),
    ]

    for task, notes in openai_tasks:
        models.append({
            'task': task,
            'model': 'gpt-4o-mini',
            'provider': 'OpenAI',
            'notes': notes
        })

    # Embeddings
    models.append({
        'task': 'Embeddings',
        'model': 'text-embedding-3-small',
        'provider': 'OpenAI',
        'notes': '1536 dimensions for semantic search'
    })

    return {
        'models': models,
        'summary': f"Using {len(models)} models across 3 providers"
    }


def format_models_text(info: Dict[str, Any]) -> str:
    """Format model info as readable text."""
    lines = ["**LLM Models in Use:**\n"]

    # Group by provider
    by_provider = {}
    for m in info['models']:
        provider = m['provider']
        if provider not in by_provider:
            by_provider[provider] = []
        by_provider[provider].append(m)

    for provider in ['Fireworks', 'Anthropic', 'OpenAI']:
        if provider not in by_provider:
            continue

        lines.append(f"\n**{provider}:**")
        for m in by_provider[provider]:
            lines.append(f"  • {m['task']}: `{m['model']}`")

    lines.append(f"\n_{info['summary']}_")

    return '\n'.join(lines)


def handle_models_command(args: str, context: dict) -> dict:
    """Handle /models command."""
    info = get_model_info()

    return {
        'text': format_models_text(info),
        'data': info
    }


def register_models_command(registry):
    """Register the /models command."""
    registry.register(
        name='models',
        handler=handle_models_command,
        description='Show LLM models used for different tasks',
        aliases=['llms', 'model']
    )
