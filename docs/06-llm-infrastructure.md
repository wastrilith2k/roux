# LLM Infrastructure

[Back to Architecture Index](../ARCHITECTURE.md)

---

## Provider Architecture

**Source**: `src/llm/`

### Provider Interface (`provider_interface.py`)

Abstract base class defining: `generate()`, `generate_stream()`, `get_context_limit()`, `get_model_name()`.

### Resilient Provider Chain (`provider_factory.py`)

Automatic failover with exponential backoff (3 retries per provider):

```
Default chain:
1. Fireworks (kimi-k2-instruct-0905 or deepseek-v3p2)
2. Fireworks fallback model
3. DeepSeek direct
4. OpenAI gpt-4o-mini
5. Anthropic Claude (last resort)
```

`generate_sync()` provides an eventlet-safe synchronous wrapper.

### Provider Implementations

| Provider | Module | Models | Context | Notes |
|----------|--------|--------|---------|-------|
| Fireworks | `fireworks_provider.py` | kimi-k2-instruct-0905 (235B) | 262K | Primary production. Async via aiohttp, sync fallback. |
| Anthropic | `anthropic_provider.py` | claude-opus-4, claude-sonnet-4.5, claude-haiku-4.5 | 200K | Tool calling support. Last-resort fallback. |
| OpenAI | `openai_provider.py` | gpt-4o, gpt-4o-mini | 128K | Embeddings, tool routing, streaming. |
| DeepSeek | (via provider factory) | deepseek-v3 | 128K | Direct API fallback. |
| OpenRouter | (via provider factory) | Various | Varies | Free tier for development. |

### Vision (`llm/vision.py`)

Multi-modal image description. Hierarchy:
1. Fireworks: Qwen3-VL-8B or Qwen2.5-VL-7B (cheap/fast)
2. OpenAI fallback: gpt-4o-mini vision

## LLM Usage by Subsystem

| Subsystem | Model | Temperature | Purpose |
|-----------|-------|-------------|---------|
| Message analysis | Provider chain | 0.3 | Mode + inner monologue |
| Main response | Provider chain | 0.7-0.8 | Response generation |
| Tool routing | gpt-4o-mini | Varies | Agentic tool decisions |
| Claim extraction | Fireworks | Varies | Post-gen verification |
| Quality critique | Provider chain | 0.2 | Response scoring |
| Image intent | Fireworks | Varies | Image classification |
| Fact extraction | Kimi K2 | 0.1 | SPO extraction |
| Importance scoring | Kimi K2 | 0.1 | 1-10 scoring |
| Biography synthesis | Kimi K2 | 0.3 | Theme paragraphs |
| Observation compression | gpt-4o-mini | 0.1 | 5-10x compression |
| Reflection | Kimi K2 | 0.3 | Daily/weekly insights |
| Goal formation | Provider chain | 0.3 | Goal creation |
| Goal decomposition | Kimi K2 direct | 0.3 | Step planning |
| Reach-out decision | Kimi K2 | 0.2 | Should-reach-out |
| Relationship extraction | Kimi K2 | 0.1 | Typed relationships |
