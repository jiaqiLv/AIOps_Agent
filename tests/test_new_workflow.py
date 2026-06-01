"""Tests for the Plan-Execute architecture.

These tests cover:
1. Plan-Execute supervisor graph structure
2. Detection agent compilation and output
3. Diagnose agent compilation and structured results
4. Subgraph registry and adapters
5. Plan parsing and routing logic
6. Prompt template formatting
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestSupervisorAgent:
    """Test Plan-Execute Supervisor Agent"""

    def test_supervisor_graph_structure(self):
        """Verify supervisor has planner, executor, finalize nodes."""
        from app.agents.supervisor_plan_execute import plan_execute_agent

        graph = plan_execute_agent.get_graph()
        nodes = list(graph.nodes.keys())

        assert "planner" in nodes, "Supervisor must have planner node"
        assert "executor" in nodes, "Supervisor must have executor node"
        assert "finalize" in nodes, "Supervisor must have finalize node"
        assert "direct_reply" in nodes, "Supervisor must have direct_reply node"

    def test_supervisor_state_schema(self):
        """Verify PlanExecuteState has required fields."""
        from app.models.plan_execute_state import PlanExecuteState

        schema = PlanExecuteState.__annotations__
        assert "messages" in schema
        assert "plan" in schema
        assert "step_results" in schema
        assert "current_step_index" in schema
        assert "plan_reasoning" in schema
        assert "final_response" in schema


class TestDetectionAgent:
    """Test Detection Agent compilation"""

    def test_detection_graph_structure(self):
        """Verify detection agent has expected nodes."""
        with patch('app.config.model_config.get_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_llm.bind_tools.return_value = mock_llm
            mock_get_llm.return_value = mock_llm

            from app.agents.detection_agent import build_detection_agent
            graph = build_detection_agent()
            g = graph.get_graph()
            nodes = list(g.nodes.keys())

            assert "model" in nodes
            assert "tools" in nodes
            assert "extract_results" in nodes
            assert "final" in nodes

    def test_detection_state_schema(self):
        """Verify DetectionAgentState has required fields."""
        from app.models.detection_agent_state import DetectionAgentState

        schema = DetectionAgentState.__annotations__
        assert "three_sigma_result" in schema
        assert "csv_file_path" in schema
        assert "inject_time" in schema
        assert "abnormal_kpi" in schema
        assert "final_response" in schema


class TestDiagnoseAgent:
    """Test Diagnose Agent compilation and structured results"""

    def test_diagnose_final_returns_structured_results(self):
        """Verify diagnose final node produces structured output (no LLM call)."""
        from app.agents.diagnose_agent import _create_diagnose_structured_final_node

        final_node = _create_diagnose_structured_final_node()

        state = {
            "rcd_result": {"success": True, "root_causes": ["metric_a"]},
            "pc_result": {"success": True, "root_causes": ["metric_b"], "edges": [["metric_b", "metric_c"]]},
            "csv_file_path": "data/test.csv",
            "inject_time": 100.0,
            "abnormal_kpi": "metric_a",
            "graph_visualizations": [],
            "tool_errors": [],
            "messages": [],
        }

        result = final_node(state)

        assert result["final_response"] is not None
        assert result["integrated_result"] is not None

        # Structured output — no LLM call
        assert "metric_a" in result["final_response"]
        assert result["fault_type"] is not None  # should be inferred from metric names
        assert result["root_causes"] is not None
        assert len(result["root_causes"]) >= 1
        assert result["propagation_path"] is not None

        # integrated_result should be JSON with structured data
        integrated = json.loads(result["integrated_result"])
        assert isinstance(integrated, dict)
        assert integrated["rcd_result"] == state["rcd_result"]
        assert integrated["pc_result"] == state["pc_result"]
        assert integrated["csv_file_path"] == state["csv_file_path"]


class TestSubgraphRegistry:
    """Test subgraph registry and adapters"""

    def test_detection_adapter_build_input(self):
        """Verify DetectionAdapter builds correct input."""
        from app.agents.subgraph_registry import get_adapter

        adapter = get_adapter("detection")
        step_input = {"task_description": "Test", "data_path": "data/test.csv"}
        result = adapter.build_input(step_input, {})

        assert result["task_description"] == "Test"
        assert result["csv_file_path"] == "data/test.csv"
        assert result["max_iterations"] == 5

    def test_diagnose_adapter_build_input_with_detection(self):
        """Verify DiagnoseAdapter enriches task with detection context."""
        from app.agents.subgraph_registry import get_adapter

        adapter = get_adapter("diagnose")
        step_input = {"task_description": "Root cause", "from_step": 1}
        step_results = {
            1: {
                "anomaly_report": [
                    {"metric": "cpu_usage", "max_z_score": 5.0},
                    {"metric": "mem_usage", "max_z_score": 3.0},
                ],
                "csv_file_path": "data/test.csv"
            }
        }
        result = adapter.build_input(step_input, step_results)

        assert "cpu_usage" in result["task_description"]
        assert "异常检测结果" in result["task_description"]
        assert result["csv_file_path"] == "data/test.csv"

    def test_diagnose_adapter_build_input_without_detection(self):
        """Verify DiagnoseAdapter works without prior detection."""
        from app.agents.subgraph_registry import get_adapter

        adapter = get_adapter("diagnose")
        step_input = {
            "task_description": "Root cause",
            "data_path": "data/test.csv",
            "inject_time": 100.0,
        }
        result = adapter.build_input(step_input, {})

        assert result["task_description"] == "Root cause"
        assert result["csv_file_path"] == "data/test.csv"
        assert result["inject_time"] == 100.0

    def test_registry_has_both_adapters(self):
        """Verify registry contains both detection and diagnose."""
        from app.agents.subgraph_registry import REGISTRY

        assert "detection" in REGISTRY
        assert "diagnose" in REGISTRY


class TestPromptTemplate:
    """Test prompt template utilities"""

    def test_format_detection_summary_with_data(self):
        """Test detection summary formatting."""
        from app.utils.prompt_template import format_detection_summary

        detection_result = {
            "success": True,
            "summary": "3-Sigma detection: found 5 anomalies",
            "csv_file_path": "data/test.csv",
            "inject_time": 100.0,
            "abnormal_kpi": "cpu_usage",
        }

        summary = format_detection_summary(detection_result)
        assert "3-Sigma detection" in summary
        assert "5 anomalies" in summary

    def test_format_detection_summary_none(self):
        """Test detection summary with None input."""
        from app.utils.prompt_template import format_detection_summary

        summary = format_detection_summary(None)
        assert "未执行" in summary

    def test_format_diagnose_summary_with_data(self):
        """Test diagnose summary formatting."""
        from app.utils.prompt_template import format_diagnose_summary

        diagnose_result = {
            "rcd_result": {"success": True, "root_causes": ["metric_a"]},
            "pc_result": {"success": True, "root_causes": ["metric_b"], "edges": [["metric_b", "metric_c"]]},
            "csv_file_path": "data/test.csv",
            "inject_time": 100.0,
            "abnormal_kpi": "metric_a",
        }

        summary = format_diagnose_summary(diagnose_result)
        assert "IAF-RCL" in summary
        assert "KE-FPC" in summary
        assert "metric_a" in summary

    def test_format_diagnose_summary_none(self):
        """Test diagnose summary with None input."""
        from app.utils.prompt_template import format_diagnose_summary

        summary = format_diagnose_summary(None)
        assert "未执行" in summary


class TestReactNodes:
    """Test generic ReAct node functions"""

    def test_route_after_model_with_tool_calls(self):
        """Test routing when LLM makes tool calls."""
        from app.agents.nodes.react_nodes import route_after_model
        from langchain_core.messages import AIMessage

        mock_msg = Mock(spec=AIMessage)
        mock_msg.tool_calls = [{"name": "test_tool"}]

        state = {
            "messages": [mock_msg],
            "iteration_count": 1,
            "max_iterations": 10,
        }

        result = route_after_model(state)
        assert result == "tools"

    def test_route_after_model_no_tool_calls(self):
        """Test routing when LLM makes no tool calls."""
        from app.agents.nodes.react_nodes import route_after_model
        from langchain_core.messages import AIMessage

        mock_msg = Mock(spec=AIMessage)
        mock_msg.tool_calls = []

        state = {
            "messages": [mock_msg],
            "iteration_count": 1,
            "max_iterations": 10,
        }

        result = route_after_model(state)
        assert result == "final"

    def test_route_after_extract_interrupt(self):
        """Test routing when interrupt is requested."""
        from app.agents.nodes.react_nodes import route_after_extract

        state = {
            "interrupted": True,
            "interrupt_data": {"question": "Need file path"},
            "messages": [],
        }

        result = route_after_extract(state)
        assert result == "interrupt"

    def test_route_after_extract_continue(self):
        """Test routing continues loop when more tools needed."""
        from app.agents.nodes.react_nodes import route_after_extract
        from langchain_core.messages import AIMessage

        mock_msg = Mock(spec=AIMessage)
        mock_msg.tool_calls = [{"name": "test_tool"}]

        state = {
            "interrupted": False,
            "messages": [Mock(), mock_msg],
            "iteration_count": 1,
            "max_iterations": 10,
        }

        result = route_after_extract(state)
        assert result == "model"


class TestModelsInit:
    """Test models __init__ exports"""

    def test_models_init_exports_new_states(self):
        """Verify models __init__ exports PlanExecuteState."""
        from app.models import PlanExecuteState, PlanStep, DetectionAgentState, ReactAgentState

        assert PlanExecuteState is not None
        assert PlanStep is not None
        assert DetectionAgentState is not None
        assert ReactAgentState is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
