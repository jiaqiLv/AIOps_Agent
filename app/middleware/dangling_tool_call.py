"""M3: Dangling Tool-Call Middleware

Repairs message histories where an ``AIMessage`` carries ``tool_calls``
that have no corresponding ``ToolMessage``.  This can happen after an
interrupt or crash and would cause LangGraph to raise a state-error.

Integration point:
  - react_nodes.create_model_node  →  called before llm.invoke()
"""

import json
from typing import List

from langchain_core.messages import AIMessage, ToolMessage

from app.utils.logger import get_logger

logger = get_logger(__name__)


def fix_dangling_tool_calls(messages: List) -> List:
    """Scan *messages* for incomplete tool_call sequences and patch them.

    For every ``tool_call_id`` in the last AIMessage that lacks a matching
    ``ToolMessage``, a placeholder error ToolMessage is appended.

    Returns:
        A (possibly extended) copy of *messages*.
    """
    if not messages:
        return messages

    # Collect all tool_call_ids that already have ToolMessages
    answered_ids = set()
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.tool_call_id:
            answered_ids.add(msg.tool_call_id)

    # Walk backwards: find the last AIMessage with unanswered tool_calls
    patched = list(messages)
    for msg in reversed(patched):
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            dangling = [
                tc for tc in msg.tool_calls
                if tc.get("id") and tc.get("id") not in answered_ids
            ]
            if dangling:
                logger.warning(
                    f"DANGLING: {len(dangling)} unanswered tool_calls detected, "
                    f"inserting placeholder ToolMessages"
                )
                for tc in dangling:
                    patched.append(ToolMessage(
                        content=json.dumps({
                            "error": "工具调用被中断或超时，已自动补全。",
                            "success": False,
                        }, ensure_ascii=False),
                        tool_call_id=tc["id"],
                        name=tc.get("name", "unknown"),
                        status="error",
                    ))
            break  # only check the last AIMessage

    return patched
