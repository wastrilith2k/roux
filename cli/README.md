# Companion CLI

Minimal terminal interface for chatting with the companion.

## Quick Start

```bash
# Build and run
./run.sh --build

# Just run (if already built)
./run.sh
```

## Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/clear` | Clear screen |
| `/status` | Show connection status |
| `/history [n]` | Show last n messages |
| `/mood` | Show companion's current mood |
| `/timestamps` | Toggle timestamp display |
| `/reconnect` | Reconnect to backend |
| `/quit` | Exit |

## Environment Variables

- `COMPANION_BACKEND_URL` - Backend URL (default: `http://agent-service:5000`)
- `COMPANION_EMAIL` - Auth email (default: `user@example.com`)
- `COMPANION_API_KEY` - API key (optional, overrides email auth)
- `NO_COLOR` - Disable colors
- `FORCE_COLOR` - Force colors even in non-TTY

## Running Locally

```bash
pip install -r requirements.txt
COMPANION_BACKEND_URL=http://localhost:5000 python3 companion_cli.py
```

## Troubleshooting

### Connection refused
Make sure the backend is running and accessible.

### Colors not showing
Set `FORCE_COLOR=1` environment variable.
