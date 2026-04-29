# Slash Commands

Slash commands are entered directly in the chat input. Commands prefixed with **[CLI]** are only available in the terminal client (`cli/companion_cli.py`); all others work in both the web chat and CLI.

---

## General

| Command | Aliases | Description |
|---------|---------|-------------|
| `/help` | `/?`, `/commands` | List all available commands |
| `/status` | `/sys`, `/info` | Show system status overview (LLM providers, DB, task queue) |
| `/models` | `/llms`, `/model` | Show which LLM is used for each task (generation, embeddings, tools) |
| `/test <message>` | — | Send a message in dry-run mode — pipeline runs but nothing is saved to the database |

---

## Memory & Knowledge

| Command | Aliases | Description |
|---------|---------|-------------|
| `/facts [n] [search]` | `/f`, `/fact` | Browse the fact store. `/facts 20` shows last 20. `/facts hiking` searches by keyword |
| `/goals [all]` | `/g`, `/goal` | View active companion goals with progress bars. `/goals all` includes completed |
| `/opinions [topic]` | `/op`, `/opinion` | View companion opinions grouped by category. `/opinions music` filters by topic |
| `/episodes [n]` | `/ep`, `/eps` | View recent conversation episodes with emotional arc. Default: last 8 |
| `/curiosity` | `/curious`, `/threads` | View active curiosity threads, sorted by urgency |
| `/biographies` | `/bio`, `/bios` | List all synthesized companion biographies |
| `/biographies refresh` | | Trigger regeneration of all biographies |
| `/biographies <name>` | | Show biography for a specific person |
| `/relationships` | `/rels`, `/rel` | View structured entity relationships from the knowledge graph |

---

## Companion State [CLI]

| Command | Aliases | Description |
|---------|---------|-------------|
| `/state` | — | Show the companion's full internal state: mood, energy, scene, activity, schedule |
| `/see` | — | Simplified scene view — what the companion looks like right now (clothing, posture, setting) |
| `/mood` | — | Show the companion's current mood label |
| `/activity` | — | Show current activity and today's schedule |
| `/autopilot` | `/where`, `/james` | Show the primary user's inferred current activity (autopilot routine model) |

---

## Fact Approval [CLI]

Roux extracts facts from conversations and queues them for approval before they enter the permanent fact store.

| Command | Aliases | Description |
|---------|---------|-------------|
| `/pending` | — | List all facts awaiting approval |
| `/approve [id]` | — | Approve a fact. Omit `id` to approve the currently displayed pending fact |
| `/reject [id] [reason]` | — | Reject a fact, with an optional reason |
| `/edit [id] <new text>` | — | Edit a pending fact and approve it. Omit `id` to edit the current fact |

---

## Image Management [CLI]

| Command | Aliases | Description |
|---------|---------|-------------|
| `/img [count]` | `/images` | Show URLs of recently generated images. Default: last 20 |

---

## History & Display [CLI]

| Command | Aliases | Description |
|---------|---------|-------------|
| `/history [n]` | `/hist` | Show the last `n` messages in the local session history. Default: 10 |
| `/clear` | `/cls` | Clear the terminal screen |
| `/timestamps` | `/ts` | Toggle timestamp display on/off |
| `/typewriter [on\|off\|<ms>]` | `/tw` | Toggle typewriter effect or set speed in ms per character. `/tw 50` = 50ms/char |

---

## Session [CLI]

| Command | Aliases | Description |
|---------|---------|-------------|
| `/reconnect` | — | Disconnect and reconnect to the backend |
| `/quit` | `/exit`, `/q` | Exit the CLI |

---

## How commands are routed

The backend uses a central `CommandRegistry` (`src/core/commands/registry.py`). When the web chat or CLI sends a message starting with `/`, it checks the registry first. Recognized commands return a `command_response` event rather than going through the conversation pipeline.

The CLI has an additional local `CommandHandler` (`cli/commands.py`) that intercepts display/session commands (clear, history, typewriter, quit, etc.) before they reach the backend.

To register a new backend command, add a file to `src/core/commands/` and call `registry.register()` from `_register_all_commands()` in `registry.py`.
