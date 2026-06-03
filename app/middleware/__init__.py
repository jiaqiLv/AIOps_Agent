"""Middleware system for AIOps Agent.

Provides cross-cutting concerns integrated at key points in the
LangGraph execution pipeline:

  M1  LLMRetryHandler         — retry + circuit breaker for LLM calls
  M2  SafeToolNode             — tool exception → error ToolMessage
  M3  fix_dangling_tool_calls  — repair incomplete tool_call sequences
  M4  detect_loop              — detect and break repetitive tool loops
  M5  summarize_if_needed      — truncate context when approaching limits
  M6  TokenUsageTracker        — log LLM token consumption
  M7  SessionDataManager       — isolated output directories per session
"""

from app.middleware.llm_error_handling import (
    LLMRetryHandler,
    get_llm_retry_handler,
    LLMCircuitBreakerError,
    LLMMaxRetriesError,
)
from app.middleware.tool_error_handling import create_safe_tool_node
from app.middleware.dangling_tool_call import fix_dangling_tool_calls
from app.middleware.loop_detection import detect_loop_from_messages
from app.middleware.summarization import summarize_if_needed
from app.middleware.token_usage import TokenUsageTracker, get_token_tracker
from app.middleware.session_data import SessionDataManager, get_session_manager

__all__ = [
    "LLMRetryHandler",
    "get_llm_retry_handler",
    "LLMCircuitBreakerError",
    "LLMMaxRetriesError",
    "create_safe_tool_node",
    "fix_dangling_tool_calls",
    "detect_loop_from_messages",
    "summarize_if_needed",
    "TokenUsageTracker",
    "get_token_tracker",
    "SessionDataManager",
    "get_session_manager",
]
