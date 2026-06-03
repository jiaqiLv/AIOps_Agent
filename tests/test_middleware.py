"""Unit tests for middleware modules M1-M7."""

import json
import pytest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage


# ═══════════════════════════════════════════════════════════════════
# M1: LLM Error Handling
# ═══════════════════════════════════════════════════════════════════

class TestLLMRetryHandler:
    """Tests for app.middleware.llm_error_handling.LLMRetryHandler"""

    def _make_handler(self, **overrides):
        from app.middleware.llm_error_handling import LLMRetryHandler
        defaults = dict(max_retries=2, base_delay=0.01, max_delay=0.05,
                        circuit_breaker_threshold=3, circuit_breaker_reset=1.0)
        defaults.update(overrides)
        return LLMRetryHandler(**defaults)

    def test_success_first_try(self):
        handler = self._make_handler()
        llm = MagicMock()
        llm.invoke.return_value = AIMessage(content="ok")
        result = handler.invoke(llm, [])
        assert result.content == "ok"
        assert llm.invoke.call_count == 1

    def test_retries_on_retryable_error(self):
        handler = self._make_handler()
        llm = MagicMock()
        llm.invoke.side_effect = [
            Exception("429 rate limit exceeded"),
            Exception("502 bad gateway"),
            AIMessage(content="ok"),
        ]
        result = handler.invoke(llm, [])
        assert result.content == "ok"
        assert llm.invoke.call_count == 3

    def test_no_retry_on_auth_error(self):
        handler = self._make_handler()
        llm = MagicMock()
        llm.invoke.side_effect = Exception("401 unauthorized")
        from app.middleware.llm_error_handling import LLMMaxRetriesError
        # Auth errors are not retryable, should raise immediately
        with pytest.raises(Exception, match="401"):
            handler.invoke(llm, [])
        assert llm.invoke.call_count == 1

    def test_max_retries_exceeded(self):
        handler = self._make_handler(max_retries=1)
        llm = MagicMock()
        llm.invoke.side_effect = Exception("500 internal server error")
        from app.middleware.llm_error_handling import LLMMaxRetriesError
        with pytest.raises(LLMMaxRetriesError):
            handler.invoke(llm, [])
        assert llm.invoke.call_count == 2  # 1 initial + 1 retry

    def test_circuit_breaker_opens(self):
        handler = self._make_handler(circuit_breaker_threshold=2)
        llm = MagicMock()
        llm.invoke.side_effect = Exception("500 server error")
        # Trigger failures
        for _ in range(2):
            try:
                handler.invoke(llm, [])
            except Exception:
                pass
        # Circuit should be open now
        from app.middleware.llm_error_handling import LLMCircuitBreakerError
        with pytest.raises(LLMCircuitBreakerError):
            handler.invoke(llm, [])

    def test_circuit_breaker_resets_on_success(self):
        handler = self._make_handler(circuit_breaker_threshold=2)
        llm = MagicMock()
        # First call succeeds
        llm.invoke.return_value = AIMessage(content="ok")
        handler.invoke(llm, [])
        # Then 2 failures
        llm.invoke.side_effect = Exception("500")
        for _ in range(2):
            try:
                handler.invoke(llm, [])
            except Exception:
                pass
        # Circuit open
        from app.middleware.llm_error_handling import LLMCircuitBreakerError
        with pytest.raises(LLMCircuitBreakerError):
            handler.invoke(llm, [])
        # Now make it succeed (circuit is half-open after reset_time)
        import time
        handler.circuit_breaker.last_failure_time = time.time() - 999
        llm.invoke.side_effect = None
        llm.invoke.return_value = AIMessage(content="ok")
        result = handler.invoke(llm, [])
        assert result.content == "ok"
        assert handler.circuit_breaker.state == "closed"


# ═══════════════════════════════════════════════════════════════════
# M2: Tool Error Handling
# ═══════════════════════════════════════════════════════════════════

class TestSafeToolNode:
    """Tests for app.middleware.tool_error_handling.create_safe_tool_node"""

    def test_catches_exception_and_returns_error_tool_messages(self):
        """Verify that an exception from the inner ToolNode is caught
        and converted to error ToolMessages."""
        from app.middleware.tool_error_handling import create_safe_tool_node

        # Patch ToolNode at the module level BEFORE create_safe_tool_node is called
        with patch("app.middleware.tool_error_handling.ToolNode") as MockToolNode:
            mock_inner = MagicMock()
            mock_inner.invoke.side_effect = RuntimeError("Tool crashed")
            MockToolNode.return_value = mock_inner

            tool = MagicMock()
            safe_node = create_safe_tool_node([tool])

            state = {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[{"id": "tc1", "name": "failing_tool", "args": {}}],
                    ),
                ]
            }
            result = safe_node(state)
            assert "messages" in result
            assert len(result["messages"]) == 1
            assert isinstance(result["messages"][0], ToolMessage)
            assert result["messages"][0].status == "error"
            error_data = json.loads(result["messages"][0].content)
            assert error_data["success"] is False

    def test_passes_through_on_success(self):
        """Verify normal ToolNode output passes through unchanged."""
        from app.middleware.tool_error_handling import create_safe_tool_node

        expected = {"messages": [ToolMessage(content="ok", tool_call_id="tc1", name="t1")]}
        with patch("app.middleware.tool_error_handling.ToolNode") as MockToolNode:
            mock_inner = MagicMock()
            mock_inner.invoke.return_value = expected
            MockToolNode.return_value = mock_inner

            tool = MagicMock()
            safe_node = create_safe_tool_node([tool])

            state = {
                "messages": [
                    AIMessage(content="", tool_calls=[{"id": "tc1", "name": "t1", "args": {}}]),
                ]
            }
            result = safe_node(state)
            assert result == expected


# ═══════════════════════════════════════════════════════════════════
# M3: Dangling Tool Call Fix
# ═══════════════════════════════════════════════════════════════════

class TestDanglingToolCallFix:
    """Tests for app.middleware.dangling_tool_call.fix_dangling_tool_calls"""

    def test_no_dangling_calls(self):
        from app.middleware.dangling_tool_call import fix_dangling_tool_calls
        messages = [
            AIMessage(content="", tool_calls=[{"id": "tc1", "name": "t1", "args": {}}]),
            ToolMessage(content="ok", tool_call_id="tc1", name="t1"),
        ]
        result = fix_dangling_tool_calls(messages)
        assert len(result) == 2

    def test_fixes_dangling_calls(self):
        from app.middleware.dangling_tool_call import fix_dangling_tool_calls
        messages = [
            AIMessage(content="", tool_calls=[
                {"id": "tc1", "name": "t1", "args": {}},
                {"id": "tc2", "name": "t2", "args": {}},
            ]),
            ToolMessage(content="ok", tool_call_id="tc1", name="t1"),
            # tc2 is missing!
        ]
        result = fix_dangling_tool_calls(messages)
        assert len(result) == 3
        assert isinstance(result[2], ToolMessage)
        assert result[2].tool_call_id == "tc2"
        assert result[2].status == "error"
        error_data = json.loads(result[2].content)
        assert error_data["success"] is False

    def test_empty_messages(self):
        from app.middleware.dangling_tool_call import fix_dangling_tool_calls
        assert fix_dangling_tool_calls([]) == []

    def test_no_tool_calls(self):
        from app.middleware.dangling_tool_call import fix_dangling_tool_calls
        messages = [HumanMessage(content="hello")]
        result = fix_dangling_tool_calls(messages)
        assert len(result) == 1


# ═══════════════════════════════════════════════════════════════════
# M4: Loop Detection
# ═══════════════════════════════════════════════════════════════════

class TestLoopDetection:
    """Tests for app.middleware.loop_detection.detect_loop_from_messages"""

    def test_no_loop(self):
        from app.middleware.loop_detection import detect_loop_from_messages
        messages = [
            AIMessage(content="", tool_calls=[{"name": "t1", "args": {"a": 1}, "id": "1"}]),
            AIMessage(content="", tool_calls=[{"name": "t2", "args": {"b": 2}, "id": "2"}]),
        ]
        assert detect_loop_from_messages(messages) is None

    def test_warning_threshold(self):
        from app.middleware.loop_detection import detect_loop_from_messages
        msgs = []
        for i in range(3):
            msgs.append(AIMessage(
                content="", tool_calls=[{"name": "same_tool", "args": {"x": 1}, "id": f"tc{i}"}]
            ))
        assert detect_loop_from_messages(msgs, warning_threshold=3, stop_threshold=5) == "warning"

    def test_stop_threshold(self):
        from app.middleware.loop_detection import detect_loop_from_messages
        msgs = []
        for i in range(5):
            msgs.append(AIMessage(
                content="", tool_calls=[{"name": "same_tool", "args": {"x": 1}, "id": f"tc{i}"}]
            ))
        assert detect_loop_from_messages(msgs, warning_threshold=3, stop_threshold=5) == "stop"

    def test_mixed_calls_no_loop(self):
        from app.middleware.loop_detection import detect_loop_from_messages
        msgs = []
        for i in range(4):
            msgs.append(AIMessage(
                content="",
                tool_calls=[{"name": f"tool_{i % 2}", "args": {}, "id": f"tc{i}"}],
            ))
        assert detect_loop_from_messages(msgs) is None

    def test_too_few_messages(self):
        from app.middleware.loop_detection import detect_loop_from_messages
        msgs = [AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}])]
        assert detect_loop_from_messages(msgs) is None


# ═══════════════════════════════════════════════════════════════════
# M5: Summarization
# ═══════════════════════════════════════════════════════════════════

class TestSummarization:
    """Tests for app.middleware.summarization.summarize_if_needed"""

    def test_no_compression_below_threshold(self):
        from app.middleware.summarization import summarize_if_needed
        messages = [
            SystemMessage(content="system"),
            HumanMessage(content="hello"),
            AIMessage(content="hi"),
        ]
        result = summarize_if_needed(messages, token_threshold=100000)
        assert len(result) == 3

    def test_truncates_when_exceeds_threshold(self):
        from app.middleware.summarization import summarize_if_needed
        # Create messages with large ToolMessage content
        messages = [
            SystemMessage(content="system"),
        ]
        for i in range(20):
            messages.append(ToolMessage(
                content="x" * 5000,  # ~1250 tokens each
                tool_call_id=f"tc{i}",
                name=f"tool_{i}",
            ))
        messages.append(HumanMessage(content="latest"))
        messages.append(AIMessage(content="response"))

        result = summarize_if_needed(messages, token_threshold=5000, keep_recent=3)
        # Should keep system + truncated middle + last 3
        assert len(result) > 0
        # First message is still system
        assert isinstance(result[0], SystemMessage)
        # Last 3 should be intact
        assert isinstance(result[-3], ToolMessage)
        assert isinstance(result[-2], HumanMessage)
        assert isinstance(result[-1], AIMessage)

    def test_preserves_recent_messages(self):
        from app.middleware.summarization import summarize_if_needed
        messages = [
            SystemMessage(content="s"),
            ToolMessage(content="a" * 20000, tool_call_id="1", name="t1"),
            ToolMessage(content="b" * 20000, tool_call_id="2", name="t2"),
            HumanMessage(content="important"),
            AIMessage(content="final"),
        ]
        result = summarize_if_needed(messages, token_threshold=1000, keep_recent=2)
        # Last 2 (HumanMessage, AIMessage) should be untouched
        assert isinstance(result[-2], HumanMessage)
        assert result[-2].content == "important"
        assert isinstance(result[-1], AIMessage)
        assert result[-1].content == "final"


# ═══════════════════════════════════════════════════════════════════
# M6: Token Usage Tracking
# ═══════════════════════════════════════════════════════════════════

class TestTokenUsage:
    """Tests for app.middleware.token_usage.TokenUsageTracker"""

    def test_track_from_usage_metadata(self):
        from app.middleware.token_usage import TokenUsageTracker
        tracker = TokenUsageTracker()
        msg = AIMessage(
            content="hello",
            usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )
        record = tracker.track("test_agent", msg)
        assert record is not None
        assert record["input_tokens"] == 100
        assert record["output_tokens"] == 50
        assert record["total_tokens"] == 150

    def test_track_no_metadata(self):
        from app.middleware.token_usage import TokenUsageTracker
        tracker = TokenUsageTracker()
        msg = AIMessage(content="hello")
        record = tracker.track("test_agent", msg)
        assert record is None

    def test_summary_aggregation(self):
        from app.middleware.token_usage import TokenUsageTracker
        tracker = TokenUsageTracker()
        tracker.track("agent_a", AIMessage(
            content="a", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        ))
        tracker.track("agent_b", AIMessage(
            content="b", usage_metadata={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}
        ))
        tracker.track("agent_a", AIMessage(
            content="c", usage_metadata={"input_tokens": 30, "output_tokens": 15, "total_tokens": 45}
        ))
        summary = tracker.summary()
        assert summary["total_calls"] == 3
        assert summary["total_input_tokens"] == 60
        assert summary["per_agent"]["agent_a"]["calls"] == 2
        assert summary["per_agent"]["agent_a"]["input"] == 40

    def test_reset(self):
        from app.middleware.token_usage import TokenUsageTracker
        tracker = TokenUsageTracker()
        tracker.track("a", AIMessage(content="x", usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}))
        tracker.reset()
        assert len(tracker.records) == 0


# ═══════════════════════════════════════════════════════════════════
# M7: Session Data
# ═══════════════════════════════════════════════════════════════════

class TestSessionData:
    """Tests for app.middleware.session_data.SessionDataManager"""

    def test_create_session(self, tmp_path):
        from app.middleware.session_data import SessionDataManager
        mgr = SessionDataManager(base_dir=str(tmp_path))
        session_dir = mgr.create_session("test_001")
        import os
        assert os.path.isdir(session_dir)
        assert "test_001" in session_dir

    def test_get_output_path(self, tmp_path):
        from app.middleware.session_data import SessionDataManager
        mgr = SessionDataManager(base_dir=str(tmp_path))
        mgr.create_session("test_002")
        path = mgr.get_output_path("report.html")
        assert path.endswith("report.html")
        assert "test_002" in path

    def test_auto_create_on_get(self, tmp_path):
        from app.middleware.session_data import SessionDataManager
        mgr = SessionDataManager(base_dir=str(tmp_path))
        path = mgr.get_output_path("auto.txt")
        import os
        assert os.path.isdir(os.path.dirname(path))

    def test_reset(self, tmp_path):
        from app.middleware.session_data import SessionDataManager
        mgr = SessionDataManager(base_dir=str(tmp_path))
        mgr.create_session("test_003")
        mgr.reset()
        assert mgr.session_id is None
        assert mgr.session_dir is None


# ═══════════════════════════════════════════════════════════════════
# Integration: create_model_node with middleware
# ═══════════════════════════════════════════════════════════════════

class TestModelNodeIntegration:
    """Test that create_model_node integrates middleware correctly."""

    def test_model_node_uses_retry_handler(self):
        from app.agents.nodes.react_nodes import create_model_node

        mock_llm = MagicMock()
        mock_llm.invoke = MagicMock(return_value=AIMessage(content="done"))
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        node = create_model_node(mock_llm, "You are a helper.")
        state = {
            "messages": [],
            "task_description": "test task",
            "iteration_count": 0,
            "max_iterations": 10,
        }
        result = node(state)
        assert "messages" in result
        # The last message should be the AI response
        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)

    def test_model_node_handles_dangling_calls(self):
        from app.agents.nodes.react_nodes import create_model_node

        mock_llm = MagicMock()
        mock_llm.invoke = MagicMock(return_value=AIMessage(content="ok"))

        node = create_model_node(mock_llm, "system")

        # State with a dangling tool_call (AIMessage has tool_calls but no ToolMessage)
        state = {
            "messages": [
                AIMessage(content="", tool_calls=[{"id": "tc1", "name": "t1", "args": {}}]),
            ],
            "iteration_count": 0,
            "max_iterations": 10,
        }
        result = node(state)
        # Should not crash; dangling calls are fixed before LLM invoke
        assert "messages" in result

    def test_model_node_handles_llm_failure(self):
        from app.agents.nodes.react_nodes import create_model_node

        mock_llm = MagicMock()
        mock_llm.invoke = MagicMock(side_effect=Exception("API error"))

        node = create_model_node(mock_llm, "system")
        state = {
            "messages": [HumanMessage(content="test")],
            "iteration_count": 0,
            "max_iterations": 10,
        }
        result = node(state)
        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert "失败" in last_msg.content or "不可用" in last_msg.content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
