"""M2: Tool Error Handling Middleware

Wraps LangGraph's ``ToolNode`` so that uncaught tool exceptions are
converted into error ``ToolMessage`` objects instead of crashing the graph.

Integration point:
  - agent_registry.build_react_agent  →  replaces ``ToolNode(tools)``
"""

import json
from typing import List, Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from app.utils.logger import get_logger

logger = get_logger(__name__)


def create_safe_tool_node(tools: List) -> Any:
    """Return a node function that executes *tools* with exception safety.

    The returned callable is a drop-in replacement for ``ToolNode(tools)``.
    If the inner ToolNode raises, every pending tool_call gets an error
    ToolMessage so the ReAct agent can react (retry, skip, or give up).
    """
    inner = ToolNode(tools)

    def safe_tool_node(state: dict) -> dict:
        try:
            return inner.invoke(state)
        except Exception as exc:
            logger.error(f"ToolNode error: {exc}")
            error_content = json.dumps(
                {"error": f"工具执行异常: {str(exc)[:500]}", "success": False},
                ensure_ascii=False,
            )
            error_msgs: List[ToolMessage] = []
            for msg in reversed(state.get("messages", [])):
                if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        error_msgs.append(
                            ToolMessage(
                                content=error_content,
                                tool_call_id=tc.get("id", ""),
                                name=tc.get("name", "unknown"),
                                status="error",
                            )
                        )
                    break
            return {"messages": error_msgs}

    return safe_tool_node
