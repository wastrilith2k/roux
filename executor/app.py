"""
Code Executor Service -- Flask API for sandboxed Python execution.

WHAT: Minimal Flask app with three endpoints: /execute (run Python code),
      /tools (list available tool modules), /health (liveness check). Code
      runs in a subprocess sandbox with configurable memory and time limits.

WHY:  The companion can write Python code to use tools (web search, Gmail,
      image generation, etc.), but executing arbitrary code in the main Flask
      process would be a security and stability risk. This service runs in a
      separate Docker container with restricted capabilities.

HOW:  POST /execute with {"code": "...", "timeout": 30} -> sandbox.execute()
      runs the code in a subprocess with resource limits. The tools/ package
      is on the Python path so `from tools import memory` works inside
      executed code. Environment variables provide API keys and DB credentials.
"""
import os
import logging
from flask import Flask, request, jsonify

from sandbox import get_sandbox

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
EXECUTOR_PORT = int(os.environ.get('EXECUTOR_PORT', 5001))
DEFAULT_TIMEOUT = int(os.environ.get('EXECUTION_TIMEOUT', 30))
MAX_MEMORY_MB = int(os.environ.get('MAX_MEMORY_MB', 512))


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'companion-code-executor'
    })


@app.route('/tools', methods=['GET'])
def list_tools():
    """List available tool modules."""
    tools_dir = '/app/tools'
    tools = []

    if os.path.exists(tools_dir):
        for filename in os.listdir(tools_dir):
            if filename.endswith('.py') and not filename.startswith('_'):
                module_name = filename[:-3]
                # Try to get module docstring
                try:
                    module_path = os.path.join(tools_dir, filename)
                    with open(module_path, 'r') as f:
                        content = f.read()
                        # Extract first docstring
                        if '"""' in content:
                            docstring = content.split('"""')[1].strip()
                        else:
                            docstring = f"Module: {module_name}"
                except Exception:
                    docstring = f"Module: {module_name}"

                tools.append({
                    'name': module_name,
                    'description': docstring.split('\n')[0]  # First line only
                })

    # Also check for README
    readme_path = os.path.join(tools_dir, 'README.md')
    readme_content = None
    if os.path.exists(readme_path):
        try:
            with open(readme_path, 'r') as f:
                readme_content = f.read()
        except Exception:
            pass

    return jsonify({
        'tools': tools,
        'readme': readme_content,
        'usage': 'from tools import <module_name>'
    })


@app.route('/execute', methods=['POST'])
def execute_code():
    """Execute Python code in sandbox.

    Request body:
        {
            "code": "python code to execute",
            "timeout": 30  # optional, default 30s
        }

    Response:
        {
            "success": true/false,
            "output": "stdout from execution",
            "error": "stderr if any",
            "return_code": 0
        }
    """
    data = request.get_json()

    if not data or 'code' not in data:
        return jsonify({
            'success': False,
            'error': 'Missing "code" in request body'
        }), 400

    code = data['code']
    timeout = data.get('timeout', DEFAULT_TIMEOUT)

    # Validate timeout
    if timeout < 1 or timeout > 120:
        return jsonify({
            'success': False,
            'error': 'Timeout must be between 1 and 120 seconds'
        }), 400

    logger.info(f"Executing code (timeout={timeout}s): {code[:100]}...")

    # Execute in sandbox
    sandbox = get_sandbox(timeout=timeout, max_memory_mb=MAX_MEMORY_MB)
    stdout, stderr, return_code = sandbox.execute(code)

    success = return_code == 0

    logger.info(f"Execution complete: success={success}, return_code={return_code}")
    if stderr:
        logger.warning(f"Stderr: {stderr[:200]}")

    return jsonify({
        'success': success,
        'output': stdout,
        'error': stderr if stderr else None,
        'return_code': return_code
    })


# ==========================================================================
# Browser endpoints — persistent Playwright session
# ==========================================================================

from browser_manager import get_browser_manager


def _browser_response(result: dict):
    """Format browser result for JSON response."""
    return jsonify(result)


def _browser_error(msg: str, status: int = 400):
    return jsonify({'error': msg}), status


@app.route('/browser/go', methods=['POST'])
def browser_go():
    """Navigate to a URL."""
    data = request.get_json() or {}
    url = data.get('url', '')
    if not url:
        return _browser_error('Missing "url"')
    try:
        result = get_browser_manager().go(url)
        return _browser_response(result)
    except Exception as e:
        return _browser_error(str(e), 500)


@app.route('/browser/click', methods=['POST'])
def browser_click():
    """Click an element by text or CSS selector."""
    data = request.get_json() or {}
    text = data.get('text', '')
    selector = data.get('selector', '')
    if not text and not selector:
        return _browser_error('Provide "text" or "selector"')
    try:
        result = get_browser_manager().click(text=text, selector=selector)
        return _browser_response(result)
    except Exception as e:
        return _browser_error(str(e), 500)


@app.route('/browser/fill', methods=['POST'])
def browser_fill():
    """Fill a form field by name or selector."""
    data = request.get_json() or {}
    value = data.get('value', '')
    name = data.get('name', '')
    selector = data.get('selector', '')
    if not value:
        return _browser_error('Missing "value"')
    if not name and not selector:
        return _browser_error('Provide "name" or "selector"')
    try:
        result = get_browser_manager().fill(value=value, name=name, selector=selector)
        return _browser_response(result)
    except Exception as e:
        return _browser_error(str(e), 500)


@app.route('/browser/submit', methods=['POST'])
def browser_submit():
    """Auto-find and click a submit button."""
    try:
        result = get_browser_manager().submit()
        return _browser_response(result)
    except Exception as e:
        return _browser_error(str(e), 500)


@app.route('/browser/type', methods=['POST'])
def browser_type():
    """Type text into a field (keystroke simulation)."""
    data = request.get_json() or {}
    text = data.get('text', '')
    name = data.get('name', '')
    selector = data.get('selector', '')
    press_enter = data.get('press_enter', False)
    if not text:
        return _browser_error('Missing "text"')
    try:
        result = get_browser_manager().type_text(text=text, name=name, selector=selector, press_enter=press_enter)
        return _browser_response(result)
    except Exception as e:
        return _browser_error(str(e), 500)


@app.route('/browser/back', methods=['POST'])
def browser_back():
    """Go back in browser history."""
    try:
        result = get_browser_manager().back()
        return _browser_response(result)
    except Exception as e:
        return _browser_error(str(e), 500)


@app.route('/browser/scroll', methods=['POST'])
def browser_scroll():
    """Scroll down or up one viewport."""
    data = request.get_json() or {}
    direction = data.get('direction', 'down')
    try:
        result = get_browser_manager().scroll(direction=direction)
        return _browser_response(result)
    except Exception as e:
        return _browser_error(str(e), 500)


@app.route('/browser/read', methods=['GET'])
def browser_read():
    """Read the current page without navigating."""
    try:
        result = get_browser_manager().read()
        return _browser_response(result)
    except Exception as e:
        return _browser_error(str(e), 500)


@app.route('/browser/close', methods=['POST'])
def browser_close():
    """Close the browser session and free resources."""
    try:
        result = get_browser_manager().close()
        return _browser_response(result)
    except Exception as e:
        return _browser_error(str(e), 500)


# ==========================================================================
# Test endpoints
# ==========================================================================

@app.route('/execute/test', methods=['GET'])
def test_execution():
    """Test endpoint to verify execution works."""
    test_code = '''
print("Hello from the companion's code executor!")
print(f"Python version: {__import__('sys').version}")

# Test tools import
try:
    from tools import memory
    print(f"Memory module loaded: {memory.__doc__[:50] if memory.__doc__ else 'No docstring'}...")
except ImportError as e:
    print(f"Could not import memory: {e}")
'''

    sandbox = get_sandbox()
    stdout, stderr, return_code = sandbox.execute(test_code)

    return jsonify({
        'success': return_code == 0,
        'output': stdout,
        'error': stderr if stderr else None,
        'test': 'basic_execution'
    })


if __name__ == '__main__':
    logger.info(f"Starting Companion Code Executor on port {EXECUTOR_PORT}")
    app.run(host='0.0.0.0', port=EXECUTOR_PORT, debug=False, threaded=True)
