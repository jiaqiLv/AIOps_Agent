"""M5: Summarization / Context Compression Middleware

When the message history grows too large (approaching the model's context
window), this middleware truncates older messages while keeping recent ones
intact so the ReAct loop can continue.

A cheap truncation strategy is used instead of an LLM-generated summary
to avoid extra token cost.

Integration point:
  - react_nodes.create_model_node  →  replaces / extends compress_messages()
"""

import json
from typing import List

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Rough estimate: 1 token ≈ 4 chars for Chinese/mixed text
_CHARS_PER_TOKEN = 4


def _estimate_tokens(messages: List) -> int:
    """Rough token count for a message list."""
    total = 0
    for msg in messages:
        if isinstance(msg.content, str):
            total += len(msg.content) // _CHARS_PER_TOKEN
        elif isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, dict):
                    total += len(str(block.get("text", ""))) // _CHARS_PER_TOKEN
                elif isinstance(block, str):
                    total += len(block) // _CHARS_PER_TOKEN
        # tool_calls add overhead
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            total += len(json.dumps(msg.tool_calls, default=str)) // _CHARS_PER_TOKEN
    return total


def summarize_if_needed(
    messages: List,
    token_threshold: int = 80000,
    keep_recent: int = 6,
) -> List:
    """Compress *messages* when they exceed *token_threshold*.

    Strategy (no extra LLM call):
      1. Always keep the first message (system prompt).
      2. Keep the last *keep_recent* messages verbatim.
      3. For messages in between, truncate large ToolMessage content.
    """
    est = _estimate_tokens(messages)
    if est < token_threshold:
        return messages

    logger.info(
        f"SUMMARIZE: Context ~{est} tokens exceeds threshold {token_threshold}, "
        f"compressing (keep_recent={keep_recent})"
    )

    if len(messages) <= keep_recent + 1:
        # Not enough messages to split; just truncate large ToolMessages
        return _truncate_large_tool_messages(messages)

    head = [messages[0]]  # system prompt
    tail = messages[-keep_recent:]
    middle = messages[1:-keep_recent]

    # Truncate large ToolMessages in the middle section
    compressed_middle = _truncate_large_tool_messages(middle, max_content_len=500)

    result = head + compressed_middle + tail
    new_est = _estimate_tokens(result)
    logger.info(f"SUMMARIZE: Compressed ~{est} → ~{new_est} tokens")
    return result


def _truncate_large_tool_messages(
    messages: List,
    max_content_len: int = 2000,
) -> List:
    """Return a copy where large ToolMessage.content is truncated."""
    result = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str) and len(msg.content) > max_content_len:
            truncated = msg.content[:max_content_len] + f"\n...[truncated, total {len(msg.content)} chars]"
            result.append(ToolMessage(
                content=truncated,
                tool_call_id=msg.tool_call_id,
                name=msg.name,
                status=msg.status,
            ))
        else:
            result.append(msg)
    return result
