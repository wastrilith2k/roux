"""
Message Bundler — Formats multiple user messages into a single pipeline request.

WHAT: When the user sends multiple messages while the companion is thinking,
      this module combines them into a single coherent message for the pipeline.
WHY:  The companion should respond to ALL pending messages in one natural reply,
      not process each one separately.
HOW:  Single messages pass through unchanged. Multiple messages get a context
      header so the companion knows to address everything in one response.
"""


def bundle_messages(messages: list) -> str:
    """
    Format multiple user messages into a single bundled message for the pipeline.

    If there's only one message, returns it as-is.
    Multiple messages get a brief context header so the companion knows
    to address all of them in one response.

    Args:
        messages: List of message strings (at least one)

    Returns:
        Single message string ready for the pipeline
    """
    if len(messages) == 1:
        return messages[0]

    parts = []
    for i, msg in enumerate(messages, 1):
        parts.append(f"({i}) {msg}")

    return (
        "[The user sent multiple messages in quick succession. "
        "Respond to all of them naturally in one reply:]\n\n"
        + "\n".join(parts)
    )
