"""M6: Token Usage Tracking Middleware

Extracts ``usage_metadata`` from LLM responses and writes structured
token-consumption records to the session TraceLogger.

Integration point:
  - react_nodes.create_model_node  →  after llm.invoke()
  - supervisor_plan_execute.planner_node  →  after llm.invoke()
"""

from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage

from app.utils.logger import get_logger

logger = get_logger(__name__)


class TokenUsageTracker:
    """Accumulates token usage per agent and writes to TraceLogger."""

    def __init__(self):
        self.records: List[Dict] = []

    def track(self, agent: str, response: Any) -> Optional[Dict]:
        """Extract token usage from *response* and record it.

        Returns the usage dict if available, else None.
        """
        usage: Optional[Dict] = None

        if isinstance(response, AIMessage):
            usage = getattr(response, "usage_metadata", None)

        if usage is None:
            # Some providers nest usage differently
            if hasattr(response, "response_metadata"):
                meta = response.response_metadata  # type: ignore
                if isinstance(meta, dict):
                    token_usage = meta.get("token_usage") or meta.get("usage")
                    if isinstance(token_usage, dict):
                        usage = token_usage

        if usage is None:
            logger.debug(f"TOKEN: No usage_metadata in response from {agent}")
            return None

        record = {
            "agent": agent,
            "input_tokens": usage.get("input_tokens") or usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("output_tokens") or usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        self.records.append(record)

        logger.info(
            f"TOKEN [{agent}]: "
            f"in={record['input_tokens']}, "
            f"out={record['output_tokens']}, "
            f"total={record['total_tokens']}"
        )

        # Also write to trace logger
        try:
            from app.utils.llm_logger import get_trace_logger
            get_trace_logger().log_tool_call(
                agent=agent,
                tool_name="_token_usage",
                args={},
                result=record,
                duration_ms=0,
            )
        except Exception:
            pass  # trace logger is optional

        return record

    def summary(self) -> Dict[str, Any]:
        """Return an aggregate summary of tracked usage."""
        per_agent: Dict[str, Dict] = {}
        total_input = total_output = 0
        for r in self.records:
            agent = r["agent"]
            if agent not in per_agent:
                per_agent[agent] = {"input": 0, "output": 0, "total": 0, "calls": 0}
            per_agent[agent]["input"] += r["input_tokens"]
            per_agent[agent]["output"] += r["output_tokens"]
            per_agent[agent]["total"] += r["total_tokens"]
            per_agent[agent]["calls"] += 1
            total_input += r["input_tokens"]
            total_output += r["output_tokens"]
        return {
            "per_agent": per_agent,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_calls": len(self.records),
        }

    def reset(self):
        self.records.clear()


# ── Singleton ───────────────────────────────────────────────────────

_tracker: Optional[TokenUsageTracker] = None


def get_token_tracker() -> TokenUsageTracker:
    global _tracker
    if _tracker is None:
        _tracker = TokenUsageTracker()
    return _tracker
