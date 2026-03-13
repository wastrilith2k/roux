#!/bin/bash
# Run Companion CLI

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PARENT_DIR"

# Container name
CONTAINER_NAME="companion-cli-persistent"

# Build if needed
if [[ "$1" == "--build" ]] || [[ "$1" == "-b" ]]; then
    docker build -t companion-cli "$SCRIPT_DIR"
    # Remove old container if rebuilding
    docker rm -f "$CONTAINER_NAME" 2>/dev/null
fi

# Check if image exists, build if not
if ! docker image inspect companion-cli >/dev/null 2>&1; then
    echo "companion-cli image not found. Building..."
    docker build -t companion-cli "$SCRIPT_DIR"
fi

# Get network name
PROJECT_NAME=$(basename "$(pwd)" | sed 's/-//g')
NETWORK_NAME="${PROJECT_NAME}_default"

# Always remove old container and create fresh (reattach is flaky)
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1

echo "Starting CLI..."
docker run -it --name "$CONTAINER_NAME" \
  --network "$NETWORK_NAME" \
  -e COMPANION_BACKEND_URL="${COMPANION_BACKEND_URL:-http://agent-service:5000}" \
  -e COMPANION_EMAIL="${COMPANION_EMAIL:-user@example.com}" \
  -e COMPANION_API_KEY="${COMPANION_API_KEY:-}" \
  -e FORCE_COLOR=1 \
  companion-cli
