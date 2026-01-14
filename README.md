# Roux

A memory library for AI agents supporting multi-user sessions, episodic memory, and conversational context.

## Features

- **Session Management**: Support for single or multi-user conversations
- **Episodic Memory**: Track individual messages and events with timestamps
- **Flexible Context**: Retrieve conversation history formatted for LLMs
- **Cost Tracking**: Built-in token usage and cost calculation

## Installation
```bash
# Clone the repo
git clone https://www.github.com/wastrilith2k/roux
cd roux

# Create virtual environment with uv
uv venv
source .venv/bin/activate

# Install in development mode
uv pip install -e .
```

## Quick Start
```python
from roux import Memory

# Create memory instance
memory = Memory()

# Create a session
session = memory.create_session(participants=['user1', 'assistant'])

# Add messages
session.add_message('user1', 'Hello!', role='user')
session.add_message('assistant', 'Hi there!', role='assistant')

# Get context for LLM
context = session.get_messages_for_llm()
```

## Running the Example Chatbot

The personal assistant chatbot demonstrates basic usage:
```bash
# Create a .env file with your OpenAI API key
echo 'OPENAI_API_KEY=your-key-here' > .env

# Run the chatbot
python examples/chatbot.py
```

## Architecture
```
Memory
├── Sessions (conversations or game sessions)
│   └── Episodes (individual messages or events)
│       ├── user_id (who created it)
│       ├── content (the message/event)
│       ├── role (for LLM formatting)
│       ├── timestamp (when it happened)
│       └── metadata (additional context)
```

## Roadmap

- [ ] Postgres storage backend (Week 2-3)
- [ ] Neo4j/Graphiti integration for temporal knowledge graphs (Week 4)
- [ ] Vector embeddings with ChromaDB (Week 5)
- [ ] Semantic memory (facts and relationships)
- [ ] Entity profiles

## Development
```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests (when we have them)
pytest
```

## Current Status

**v0.1.0** - Basic in-memory storage with session management and episodic memory.

## License

MIT