"""LLM Conversation Logger

Logs LLM input/output conversations to files for debugging.
Includes session-based TraceLogger for chronological JSONL tracing.
"""

import os
import json
import time
from datetime import datetime
from typing import List, Any, Optional, Dict
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Log directory
LOG_DIR = Path("log/llm_conversations")


def ensure_log_dir() -> Path:
    """Ensure log directory exists."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR


def message_to_dict(msg: BaseMessage) -> dict:
    """Convert a message object to a dictionary for logging."""
    result = {
        "type": type(msg).__name__,
    }

    # Add content
    if hasattr(msg, 'content'):
        content = msg.content
        # Truncate very long content for readability
        if isinstance(content, str) and len(content) > 5000:
            result["content"] = content[:5000] + f"...[truncated, total {len(content)} chars]"
        else:
            result["content"] = content

    # Add tool_calls if present
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        result["tool_calls"] = []
        for tc in msg.tool_calls:
            if isinstance(tc, dict):
                result["tool_calls"].append({
                    "name": tc.get("name"),
                    "args": tc.get("args"),
                    "id": tc.get("id")
                })
            else:
                result["tool_calls"].append({
                    "name": getattr(tc, "name", None),
                    "args": getattr(tc, "args", None),
                    "id": getattr(tc, "id", None)
                })

    # Add tool_call_id if present (for ToolMessage)
    if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
        result["tool_call_id"] = msg.tool_call_id

    return result


def log_llm_conversation(
    agent_name: str,
    iteration: int,
    input_messages: List[BaseMessage],
    response: Optional[Any] = None,
    error: Optional[str] = None,
    metadata: Optional[dict] = None
) -> str:
    """
    Log LLM conversation to a timestamped file.

    Args:
        agent_name: Name of the agent (e.g., "diagnose", "supervisor")
        iteration: Iteration number
        input_messages: List of input messages
        response: LLM response (optional)
        error: Error message if failed (optional)
        metadata: Additional metadata (optional)

    Returns:
        Path to the log file
    """
    ensure_log_dir()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Remove last 3 digits
    filename = f"{agent_name}_{timestamp}_iter{iteration}.jsonl"
    filepath = LOG_DIR / filename

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent_name,
        "iteration": iteration,
        "metadata": metadata or {},
    }

    # Log input messages
    log_entry["input"] = {
        "message_count": len(input_messages),
        "messages": [message_to_dict(msg) for msg in input_messages]
    }

    # Calculate approximate token count (rough estimate: 1 token ≈ 4 chars)
    total_chars = sum(len(str(msg.content)) for msg in input_messages if hasattr(msg, 'content'))
    log_entry["input"]["approx_tokens"] = total_chars // 4

    # Log response
    if response is not None:
        log_entry["response"] = {
            "type": type(response).__name__,
        }
        if hasattr(response, 'content'):
            content = response.content
            if isinstance(content, str) and len(content) > 10000:
                log_entry["response"]["content"] = content[:10000] + f"...[truncated, total {len(content)} chars]"
            else:
                log_entry["response"]["content"] = content
        if hasattr(response, 'tool_calls') and response.tool_calls:
            log_entry["response"]["tool_calls_count"] = len(response.tool_calls)
            log_entry["response"]["tool_calls"] = [
                {"name": tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)}
                for tc in response.tool_calls
            ]
    elif error:
        log_entry["error"] = error

    # Write to file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False, indent=2))

    logger.info(f"LLM conversation logged to: {filepath}")
    return str(filepath)


def log_tool_execution(
    agent_name: str,
    tool_name: str,
    tool_args: dict,
    result: str,
    execution_time: float,
    error: Optional[str] = None
) -> str:
    """
    Log tool execution to a timestamped file.

    Args:
        agent_name: Name of the agent
        tool_name: Name of the tool
        tool_args: Tool arguments
        result: Tool execution result
        execution_time: Execution time in seconds
        error: Error message if failed

    Returns:
        Path to the log file
    """
    ensure_log_dir()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{agent_name}_tool_{tool_name}_{timestamp}.jsonl"
    filepath = LOG_DIR / filename

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent_name,
        "tool": tool_name,
        "args": tool_args,
        "execution_time_seconds": execution_time,
    }

    # Truncate result if too long
    if len(result) > 20000:
        log_entry["result"] = result[:20000] + f"...[truncated, total {len(result)} chars]"
    else:
        log_entry["result"] = result

    if error:
        log_entry["error"] = error

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False, indent=2))

    logger.debug(f"Tool execution logged to: {filepath}")
    return str(filepath)


def get_recent_logs(agent_name: Optional[str] = None, limit: int = 10) -> List[str]:
    """
    Get paths to recent log files.

    Args:
        agent_name: Filter by agent name (optional)
        limit: Maximum number of log files to return

    Returns:
        List of log file paths
    """
    ensure_log_dir()

    pattern = f"{agent_name}_*" if agent_name else "*.jsonl"
    log_files = sorted(LOG_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

    return [str(f) for f in log_files[:limit]]


# ==================== Session-based Trace Logger ====================

TRACE_DIR = Path("log/traces")

# Truncation limit for large content (50KB)
_MAX_CONTENT_BYTES = 50 * 1024

# Global singleton
_trace_logger: Optional["TraceLogger"] = None


def _truncate_content(content: Any, max_bytes: int = _MAX_CONTENT_BYTES) -> Any:
    """Truncate string content to max_bytes, returning a summary suffix."""
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, default=str)
    if len(content.encode("utf-8", errors="replace")) <= max_bytes:
        return content
    # Truncate and add note
    truncated = content[:max_bytes]
    return truncated + f"\n...[truncated, total {len(content)} chars]"


class TraceLogger:
    """Session-based trace logger that writes chronological JSONL events.

    All LLM calls and tool calls from one run are written to a single file,
    making it easy to trace the full execution flow end-to-end.
    """

    def __init__(self, session_id: Optional[str] = None):
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = session_id
        self.filepath = TRACE_DIR / f"trace_{session_id}.jsonl"
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Trace session started: {self.filepath}")

    def _write_event(self, event: Dict) -> None:
        """Write a single JSON event to the trace file with indentation."""
        event["ts"] = datetime.now().isoformat(timespec="milliseconds")
        try:
            pretty = json.dumps(event, ensure_ascii=False, default=str, indent=2)
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(pretty + "\n\n")
        except (TypeError, ValueError) as e:
            logger.warning(f"TraceLogger: Failed to serialize event: {e}")

    def log_llm_call(
        self,
        agent: str,
        input_messages: List,
        response: Any,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Log an LLM inference call.

        Args:
            agent: Name identifying the caller (e.g. "supervisor_planner").
            input_messages: Messages sent to the LLM.
            response: LLM response object.
            metadata: Optional extra info dict.
        """
        # Serialize input messages
        input_serialized = []
        for msg in input_messages:
            if isinstance(msg, BaseMessage):
                input_serialized.append(message_to_dict(msg))
            elif isinstance(msg, dict):
                input_serialized.append(msg)
            else:
                input_serialized.append({"type": type(msg).__name__, "content": str(msg)})

        # Serialize response
        response_serialized: Dict[str, Any] = {}
        if response is not None:
            if isinstance(response, BaseMessage):
                response_serialized = message_to_dict(response)
            elif isinstance(response, str):
                response_serialized = {"type": "str", "content": _truncate_content(response)}
            else:
                response_serialized = {"type": type(response).__name__, "content": _truncate_content(str(response))}

        self._write_event({
            "type": "llm_call",
            "agent": agent,
            "input_messages": input_serialized,
            "response": response_serialized,
            "metadata": metadata or {},
        })

    def log_tool_call(
        self,
        agent: str,
        tool_name: str,
        args: Dict,
        result: Any,
        duration_ms: float,
        error: Optional[str] = None,
    ) -> None:
        """Log a tool execution call.

        Args:
            agent: Name of the agent that invoked the tool.
            tool_name: Name of the tool.
            args: Tool arguments dict.
            result: Tool return value (string or dict).
            duration_ms: Execution time in milliseconds.
            error: Error message if the tool failed.
        """
        result_str = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
        self._write_event({
            "type": "tool_call",
            "agent": agent,
            "tool": tool_name,
            "args": args,
            "result": _truncate_content(result_str),
            "duration_ms": round(duration_ms, 1),
            "error": error,
        })


def get_trace_logger() -> TraceLogger:
    """Get or create the global TraceLogger singleton."""
    global _trace_logger
    if _trace_logger is None:
        _trace_logger = TraceLogger()
    return _trace_logger


def reset_trace_logger(session_id: Optional[str] = None) -> TraceLogger:
    """Create a fresh TraceLogger for a new run. Call at the start of each request."""
    global _trace_logger
    _trace_logger = TraceLogger(session_id=session_id)
    return _trace_logger
