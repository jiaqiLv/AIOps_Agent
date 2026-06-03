"""M4: Loop Detection Middleware

Detects when a ReAct agent repeats the same tool calls without making
progress and either injects a warning or forces termination.

Integration points:
  - react_nodes.create_model_node  →  post-processing on LLM response
  - react_nodes.route_after_model  →  force routing to "final"
"""

import hashlib
import json
from typing import List, Optional, Dict

from langchain_core.messages import AIMessage

from app.utils.logger import get_logger

logger = get_logger(__name__)


def _call_signature(tool_calls: List[Dict]) -> str:
    """Create a stable hash from a list of tool_calls for comparison."""
    normalized = sorted(
        [{"name": tc.get("name", ""), "args": tc.get("args", {})} for tc in tool_calls],
        key=lambda x: x["name"],
    )
    raw = json.dumps(normalized, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()


def detect_loop_from_messages(
    messages: List,
    warning_threshold: int = 3,
    stop_threshold: int = 5,
) -> Optional[str]:
    """Analyse *messages* for repetitive tool-call patterns.

    Returns:
        ``None``  – no loop detected
        ``"warning"`` – same tool_call pattern repeated ≥ *warning_threshold*
        ``"stop"`` – same tool_call pattern repeated ≥ *stop_threshold*
    """
    # Extract tool-call signatures from all AIMessages
    call_sigs: List[str] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            call_sigs.append(_call_signature(msg.tool_calls))

    if len(call_sigs) < warning_threshold:
        return None

    # Count consecutive identical calls from the tail
    last_sig = call_sigs[-1]
    identical = 0
    for sig in reversed(call_sigs):
        if sig == last_sig:
            identical += 1
        else:
            break

    if identical >= stop_threshold:
        logger.warning(f"LOOP: Hard stop – {identical} identical consecutive calls")
        return "stop"
    if identical >= warning_threshold:
        logger.warning(f"LOOP: Warning – {identical} identical consecutive calls")
        return "warning"

    return None
