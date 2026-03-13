#!/usr/bin/env python3
"""
Safe Docker Compose Wrapper
Prevents accidental destructive operations on production
"""
import sys
import os
from pathlib import Path

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.environment import (
    is_production,
    get_environment_name,
    get_compose_file_path,
    require_confirmation,
)


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: safe_docker_compose.py <docker-compose-command>")
        print("\nSafe wrapper for docker-compose that prevents mistakes on production.")
        print(f"\nCurrent environment: {get_environment_name()}")
        print(f"Compose file: {get_compose_file_path()}")
        sys.exit(1)

    # Get the command
    command = sys.argv[1]
    rest_args = sys.argv[2:] if len(sys.argv) > 2 else []

    # Check if this is a potentially dangerous command
    dangerous_commands = {
        "down": "Stop all services",
        "rm": "Remove containers",
        "pull": "Pull new images (may restart services)",
        "restart": "Restart services",
        "restart": "Restart one or more services",
        "up": "Start all services",
        "build": "Rebuild images",
    }

    is_prod = is_production()

    if command in dangerous_commands and is_prod:
        action = dangerous_commands[command]
        if not require_confirmation(action, is_prod=True):
            sys.exit(1)

    # Build docker-compose command
    compose_file = str(get_compose_file_path())
    docker_cmd = f"docker compose -f {compose_file} {command} {' '.join(rest_args)}"

    print(f"\n📋 Running: {docker_cmd}\n")

    # Execute the command
    exit_code = os.system(docker_cmd)
    sys.exit(exit_code >> 8)  # Extract actual exit code


if __name__ == "__main__":
    main()
