#!/usr/bin/env python3
"""
Check current environment (production or development)
Use this before running critical commands
"""
import sys
import os

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.environment import print_environment_info, is_production


if __name__ == "__main__":
    print_environment_info()

    if is_production():
        print("🚨 YOU ARE ON PRODUCTION - BE CAREFUL!")
        print("\nAlways verify commands before running them here.")
        print("Use 'scripts/safe_docker_compose.py' for docker-compose operations.")
    else:
        print("✅ You are on development - safer to experiment")
