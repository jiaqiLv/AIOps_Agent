"""Tests for the config-driven detection agent.

Mode 1 — Direct agent invocation with mocked LLM.
Mode 2 — Supervisor-style activation via subgraph registry adapter.
"""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch


# ---------------------------------------------------------------------------
# Mode 1: Direct detection agent invocation
# ---------------------------------------------------------------------------

class TestDetectionAgentDirect:
    """Build detection agent via registry and invoke with mocked LLM."""

    @staticmethod
    def _build_with_mock_llm(llm_invoke_side_effect):
        """Patch LLM creation, build agent, return compiled graph."""
        with patch("app.config.model_config.get_llm") as mock_get_llm:
            mock_llm = Mock()
            mock_llm.bind_tools.return_value = mock_llm
            mock_llm.invoke.side_effect = llm_invoke_side_effect
            mock_get_llm.return_value = mock_llm

            from app.agents.detection_agent import build_detection_agent
            return build_detection_agent()

    def test_config_loads_from_yaml(self):
        """Verify the registry can load detection config."""
        from app.agents.agent_registry import load_agent_config
        cfg = load_agent_config("detection")
        assert cfg["name"] == "detection"
        assert "csv_reader_tool" in cfg["tools"]
        assert "three_sigma_tool" in cfg["tools"]
        # refine_prompt and termination_signal are removed — detection
        # now uses structured final node (no LLM refine)
        assert "refine_prompt" not in cfg
        assert "termination_signal" not in cfg
        assert cfg["state_schema_cls"] is not None

    def test_agent_compiles(self):
        """Agent should compile without errors."""
        agent = self._build_with_mock_llm([])
        assert agent is not None

    def test_direct_invocation_no_tool_calls(self):
        """When LLM makes no tool calls, agent goes to final node."""
        resp = Mock()
        resp.content = "I don't need tools"
        resp.tool_calls = []

        refine_resp = Mock()
        refine_resp.content = "检测摘要：未执行工具调用。"

        call_count = [0]
        def invoke_side_effect(messages):
            call_count[0] += 1
            if call_count[0] <= 1:
                return resp
            return refine_resp

        agent = self._build_with_mock_llm(invoke_side_effect)

        state = {
            "messages": [],
            "task_description": "检测异常",
            "max_iterations": 5,
            "iteration_count": 0,
            "tool_errors": [],
            "tool_results": {},
        }

        result = agent.invoke(state)
        assert result is not None
        assert result.get("final_response") is not None

    def test_termination_signal_routes_to_final(self):
        """When three_sigma_result has success=True, agent routes to final."""
        from app.agents.agent_registry import _create_route_after_extract
        route_fn = _create_route_after_extract("three_sigma_result")

        from langchain_core.messages import AIMessage
        ai_msg = AIMessage(content="", tool_calls=[{"name": "test", "args": {}, "id": "1"}])

        state = {
            "three_sigma_result": {"success": True, "anomalies": []},
            "messages": [ai_msg],
            "iteration_count": 1,
            "max_iterations": 5,
        }

        result = route_fn(state)
        assert result == "final"

    def test_termination_signal_not_triggered(self):
        """When three_sigma_result is not set, loop continues."""
        from app.agents.agent_registry import _create_route_after_extract
        route_fn = _create_route_after_extract("three_sigma_result")

        from langchain_core.messages import AIMessage
        ai_msg = AIMessage(content="", tool_calls=[{"name": "test", "args": {}, "id": "1"}])

        # three_sigma_result is None → should continue looping
        state = {
            "messages": [ai_msg],
            "iteration_count": 1,
            "max_iterations": 5,
        }

        result = route_fn(state)
        assert result == "model"

    def test_termination_signal_on_failure(self):
        """When three_sigma_result exists but failed, still terminate (no retry)."""
        from app.agents.agent_registry import _create_route_after_extract
        route_fn = _create_route_after_extract("three_sigma_result")

        from langchain_core.messages import AIMessage
        ai_msg = AIMessage(content="", tool_calls=[{"name": "test", "args": {}, "id": "1"}])

        state = {
            "three_sigma_result": {"success": False, "error": "no data"},
            "messages": [ai_msg],
            "iteration_count": 1,
            "max_iterations": 5,
        }

        result = route_fn(state)
        assert result == "final"

    def test_ensure_abnormal_kpi_sets_top_anomaly(self):
        """_ensure_abnormal_kpi picks the top anomaly metric."""
        from app.agents.agent_registry import _ensure_abnormal_kpi

        state = {
            "three_sigma_result": {
                "success": True,
                "anomalies": [
                    {"metric": "cpu_usage", "z_score": 5.0},
                    {"metric": "mem_usage", "z_score": 3.0},
                ],
            },
        }
        _ensure_abnormal_kpi(state)
        assert state["abnormal_kpi"] == "cpu_usage"

    def test_ensure_abnormal_kpi_preserves_existing(self):
        """_ensure_abnormal_kpi doesn't overwrite existing value."""
        from app.agents.agent_registry import _ensure_abnormal_kpi

        state = {
            "abnormal_kpi": "custom_kpi",
            "three_sigma_result": {
                "success": True,
                "anomalies": [{"metric": "cpu_usage", "z_score": 5.0}],
            },
        }
        _ensure_abnormal_kpi(state)
        assert state["abnormal_kpi"] == "custom_kpi"


# ---------------------------------------------------------------------------
# Mode 2: Supervisor activation via subgraph registry adapter
# ---------------------------------------------------------------------------

class TestDetectionViaAdapter:
    """Test the DetectionAdapter contract with the Plan-Execute supervisor.

    Verifies that the adapter correctly builds input and extracts results
    from the detection subgraph output.
    """

    def test_adapter_build_input(self):
        """DetectionAdapter builds correct subgraph input state."""
        from app.agents.subgraph_registry import get_adapter

        adapter = get_adapter("detection")
        step_input = {
            "task_description": "检测异常",
            "data_path": "data/test.csv",
            "inject_time": 1736039280.0,
        }
        input_state = adapter.build_input(step_input, {})

        assert input_state["task_description"] == "检测异常"
        assert input_state["csv_file_path"] == "data/test.csv"
        assert input_state["inject_time"] == 1736039280.0
        assert input_state["max_iterations"] == 5

    def test_adapter_extract_result_success(self):
        """DetectionAdapter extracts correct result on success."""
        from app.agents.subgraph_registry import get_adapter

        adapter = get_adapter("detection")
        subgraph_output = {
            "three_sigma_result": {
                "success": True,
                "anomalies": [
                    {"metric": "latency_ms", "z_score": 4.1, "value": 250.0,
                     "baseline_mean": 100.0, "baseline_std": 36.5},
                ],
                "parameters": {"threshold": 3.0},
                "metrics_checked": 5,
            },
            "csv_file_path": "data/test.csv",
            "inject_time": 1736039280.0,
            "abnormal_kpi": "latency_ms",
            "final_response": "检测到 1 个异常指标",
        }

        result = adapter.extract_result(subgraph_output)
        assert result["success"] is True
        assert result["summary"] == "检测到 1 个异常指标"
        assert result["csv_file_path"] == "data/test.csv"
        assert result["abnormal_kpi"] == "latency_ms"

    def test_adapter_extract_result_failure(self):
        """DetectionAdapter extracts correct result on failure."""
        from app.agents.subgraph_registry import get_adapter

        adapter = get_adapter("detection")
        subgraph_output = {
            "three_sigma_result": {"success": False, "error": "no data"},
            "final_response": "",
        }

        result = adapter.extract_result(subgraph_output)
        assert result["success"] is False


# ---------------------------------------------------------------------------
# Registry unit tests
# ---------------------------------------------------------------------------

class TestAgentRegistry:
    """Unit tests for the registry itself."""

    def test_load_unknown_agent_raises(self):
        from app.agents.agent_registry import load_agent_config
        with pytest.raises(ValueError, match="not found"):
            load_agent_config("nonexistent_agent")

    def test_state_schema_resolved(self):
        from app.agents.agent_registry import load_agent_config
        from app.models.detection_agent_state import DetectionAgentState
        cfg = load_agent_config("detection")
        assert cfg["state_schema_cls"] is DetectionAgentState

    def test_refine_prompt_exists(self):
        """Verify the refine prompt file is loadable."""
        from app.utils.prompt_loader import load_prompt
        content = load_prompt("app/prompts/detection_refine.md")
        assert "DETECTION_RAW_DATA" in content
        assert "TASK_DESCRIPTION" in content

    def test_fallback_detection_summary(self):
        """_fallback_detection_summary builds a readable string."""
        from app.agents.agent_registry import _fallback_detection_summary

        state = {
            "csv_file_path": "data/test.csv",
            "three_sigma_result": {
                "success": True,
                "anomalies": [
                    {"metric": "cpu", "z_score": 4.5},
                ],
            },
        }
        summary = _fallback_detection_summary(state)
        assert "3-Sigma" in summary
        assert "cpu" in summary

    def test_refine_prompt_render(self):
        """Verify template rendering with placeholders."""
        from app.utils.prompt_template import render_template
        rendered = render_template("app/prompts/detection_refine.md", {
            "DETECTION_RAW_DATA": '{"success": true}',
            "TASK_DESCRIPTION": "测试任务",
        })
        assert "测试任务" in rendered
        assert '"success": true' in rendered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
