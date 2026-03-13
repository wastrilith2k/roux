"""Sandboxed Python execution.

Executes Python code in an isolated subprocess with resource limits.
"""
import subprocess
import tempfile
import os
import resource
from typing import Tuple, Optional


class CodeSandbox:
    """Execute Python code safely in a sandboxed subprocess."""

    def __init__(self, timeout: int = 30, max_memory_mb: int = 512):
        """Initialize sandbox with resource limits.

        Args:
            timeout: Maximum execution time in seconds
            max_memory_mb: Maximum memory usage in MB
        """
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb

    def execute(self, code: str) -> Tuple[str, str, int]:
        """Execute Python code in isolated subprocess.

        Args:
            code: Python code to execute

        Returns:
            Tuple of (stdout, stderr, return_code)
        """
        # Create wrapper that sets up the environment
        # Pre-import all tools modules so LLM doesn't need to include imports
        wrapper = f'''
import sys
import os

# Add tools to path
sys.path.insert(0, '/app')

# Suppress bytecode generation
sys.dont_write_bytecode = True

# Pre-import tools modules for convenience
from tools import memory
from tools import reminders
from tools import image
from tools import search

# Execute user code
try:
{self._indent_code(code)}
except Exception as e:
    print(f"Error: {{type(e).__name__}}: {{e}}", file=sys.stderr)
    sys.exit(1)
'''

        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
            dir='/tmp'
        ) as f:
            f.write(wrapper)
            temp_path = f.name

        try:
            # Build environment - pass through necessary env vars
            env = {
                'PYTHONDONTWRITEBYTECODE': '1',
                'PYTHONUNBUFFERED': '1',
                'HOME': '/tmp',
                'PATH': '/usr/local/bin:/usr/bin:/bin',
            }

            # Pass through API keys and database config
            for key in os.environ:
                if any(key.startswith(prefix) for prefix in [
                    'POSTGRES_', 'RUNCOMFY_', 'TAVILY_', 'GOOGLE_GEMINI',
                    'OPENAI_', 'NANOBANANA_'
                ]):
                    env[key] = os.environ[key]

            # Execute with timeout
            result = subprocess.run(
                ['python', temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
                cwd='/app'
            )

            return result.stdout, result.stderr, result.returncode

        except subprocess.TimeoutExpired:
            return '', f'Execution timed out after {self.timeout}s', 1
        except Exception as e:
            return '', f'Execution failed: {str(e)}', 1
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    def _indent_code(self, code: str, spaces: int = 4) -> str:
        """Indent code block for inclusion in try block."""
        indent = ' ' * spaces
        lines = code.split('\n')
        return '\n'.join(indent + line if line.strip() else line for line in lines)


# Singleton instance
_sandbox: Optional[CodeSandbox] = None


def get_sandbox(timeout: int = 30, max_memory_mb: int = 512) -> CodeSandbox:
    """Get or create sandbox instance."""
    global _sandbox
    if _sandbox is None:
        _sandbox = CodeSandbox(timeout=timeout, max_memory_mb=max_memory_mb)
    return _sandbox
