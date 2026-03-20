# Web & CLI Interfaces

[Back to Architecture Index](../ARCHITECTURE.md)

---

## Web Chat (`public/index.html`, `public/chat.js`)

**Features**:
- Real-time WebSocket chat via Flask-SocketIO
- Typewriter effect on responses (configurable speed 5-80ms per character, click to skip)
- Companion info bar: current scene/activity, mood (with emoji), energy level, schedule
- Settings panel: auth email, companion name, typewriter speed, history count (5-50)
- Companion selector dropdown
- Message history loads on connect
- Slash command pass-through
- Mobile responsive layout

**Mood emoji mapping**: 18 moods mapped to emoji (relaxed -> relieved face, creative -> palette, etc.). Customizable with `<img>` tags for avatar images.

## CLI Chat (`cli/companion_cli.py`)

Async terminal client using `python-socketio` and `prompt_toolkit`.

**Commands**:

| Category | Commands |
|----------|----------|
| Chat | `/help`, `/clear`, `/status`, `/history`, `/mood`, `/see`, `/activity`, `/state` |
| Fact approval | `/pending`, `/approve`, `/reject`, `/edit` |
| Display | `/typewriter` (toggle/set speed), `/timestamps` |
| System | `/test` (messages don't persist), `/reconnect`, `/quit` |

Features: multi-line input (backslash + Enter), typewriter effect, thinking indicator animation.

## SocketIO Events (Server)

| Event | Direction | Description |
|-------|-----------|-------------|
| `connect` | Client -> Server | Authenticate + join room |
| `send_message` | Client -> Server | Message with bundling/cancellation |
| `disconnect` | Client -> Server | Clean up scene state |
| `get_message_history` | Client -> Server | Paginated history |
| `get_conversation_state` | Client -> Server | Relationship state, emotion, closeness |
| `response` | Server -> Client | Companion response |
| `thinking` | Server -> Client | Processing indicator |
| `image_ready` | Server -> Client | Generated image URL |
| `fact_approval_request` | Server -> Client | Pending fact for approval |
| `interjection` | Server -> Client | Unprompted companion message |
