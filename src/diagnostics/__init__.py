"""
Companion Diagnostics Package

Health checks and system monitoring.
"""

from .system_health import (
    SystemHealth,
    CheckResult,
    run_diagnostics,
    print_diagnostics,
)

__all__ = [
    'SystemHealth',
    'CheckResult',
    'run_diagnostics',
    'print_diagnostics',
]
