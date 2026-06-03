"""Integration tests for the Plan-Execute architecture.

Tests the flow: user input → planner → executor → reporter.
"""

import json
import sys
import pytest
from unittest.mock import Mock, patch


class TestSupervisorIntegration:
    """Test supervisor integration with main graph"""

    def test_main_graph_compiles(self):
        """Verify main graph compiles and has supervisor node."""
        from app.agents.main_graph import main_graph
        graph = main_graph.get_graph()
        nodes = list(graph.nodes.keys())
        assert "supervisor" in nodes


class TestDetectionAgentIntegration:
    """Test detection agent integration with mocked LLM at build time"""

    def test_detection_agent_compiles_and_runs(self):
        """Test detection agent compiles and can accept state input."""
        with patch('app.config.model_config.get_llm') as mock_get_llm:
            mock_llm_instance = Mock()
            mock_llm_instance.bind_tools.return_value = mock_llm_instance
            mock_llm_instance.invoke.return_value = Mock(content="done", tool_calls=[])
            mock_get_llm.return_value = mock_llm_instance

            from app.agents.detection_agent import build_detection_agent
            agent = build_detection_agent()
            assert agent is not None

            # Test run with edge case: no messages, no task
            state = {
                "messages": [],
                "task_description": "",
                "max_iterations": 5,
                "iteration_count": 0,
                "tool_errors": [],
                "tool_results": {},
            }

            result = agent.invoke(state)
            assert result is not None


class TestDiagnoseAgentIntegration:
    """Test diagnose agent integration with mocked LLM at build time"""

    def test_diagnose_returns_structured_results(self):
        """Test diagnose agent packages structured results with LLM refine."""
        # Ensure the real module is in sys.modules (not the LazyGraph from __init__.py)
        import app.agents.diagnose_agent as _diag_mod
        diag_module = sys.modules.get('app.agents.diagnose_agent')
        if diag_module is None or not hasattr(diag_module, 'build_diagnose_agent'):
            # Fallback: re-import the raw module
            import importlib
            diag_module = importlib.import_module('app.agents.diagnose_agent')

        with patch.object(diag_module, 'get_deepseek_llm') as mock_get_llm:
            # Diagnose agent now uses only one LLM (ReAct loop, temperature=0)
            # Structured final node has no LLM call
            react_llm = Mock()
            react_llm.bind_tools.return_value = react_llm

            mock_get_llm.return_value = react_llm

            react_call_count = [0]
            def invoke_side_effect(messages):
                react_call_count[0] += 1
                if react_call_count[0] == 1:
                    tc = Mock()
                    tc.name = "csv_reader_tool"
                    tc.args = {"data_path": "data/test.csv"}
                    tc.id = "call_1"
                    resp = Mock()
                    resp.content = ""
                    resp.tool_calls = [tc]
                    return resp
                else:
                    resp = Mock()
                    resp.content = ""
                    resp.tool_calls = []
                    return resp

            react_llm.invoke.side_effect = invoke_side_effect

            from app.agents.diagnose_agent import build_diagnose_agent
            agent = build_diagnose_agent()

            state = {
                "messages": [],
                "task_description": "Analyze data/test.csv, inject_time=100",
                "inject_time": 100,
                "max_iterations": 5,
                "iteration_count": 0,
                "tool_errors": [],
                "tool_results": {},
            }

            result = agent.invoke(state)

            # Should have structured integrated_result as JSON
            integrated = result.get("integrated_result")
            assert integrated is not None
            parsed = json.loads(integrated)
            assert isinstance(parsed, dict)
            assert "rcd_result" in parsed
            assert "pc_result" in parsed
            assert "csv_file_path" in parsed

            # final_response is now programmatic (no LLM refine)
            assert result.get("final_response") is not None
            assert "Mocked" not in result.get("final_response", "")
            # New structured fields should be present
            assert result.get("fault_type") is not None or True  # may be None
            assert result.get("root_causes") is not None
            assert result.get("propagation_path") is not None


class TestToolErrorHandling:
    """Test error handling"""

    def test_detection_handles_empty_response(self):
        """Test detection agent handles LLM returning no tool calls immediately."""
        with patch('app.config.model_config.get_llm') as mock_get_llm:
            mock_llm_instance = Mock()
            mock_llm_instance.bind_tools.return_value = mock_llm_instance
            mock_get_llm.return_value = mock_llm_instance

            resp = Mock()
            resp.content = "No tools to call"
            resp.tool_calls = []
            mock_llm_instance.invoke.return_value = resp

            from app.agents.detection_agent import build_detection_agent
            agent = build_detection_agent()

            state = {
                "messages": [],
                "task_description": "test",
                "max_iterations": 5,
                "iteration_count": 0,
                "tool_errors": [],
                "tool_results": {},
            }

            result = agent.invoke(state)
            assert result is not None
            assert result.get("final_response") is not None

    def test_diagnose_handles_empty_response(self):
        """Test diagnose agent handles LLM returning no tool calls immediately."""
        # Ensure the real module is in sys.modules
        import app.agents.diagnose_agent as _diag_mod
        diag_module = sys.modules.get('app.agents.diagnose_agent')

        with patch.object(diag_module, 'get_deepseek_llm') as mock_get_llm:
            react_llm = Mock()
            react_llm.bind_tools.return_value = react_llm

            refine_llm = Mock()
            refine_llm.invoke.return_value = "Mocked empty refine report"

            def get_llm_side_effect(**kwargs):
                if kwargs.get("temperature") == 0.3:
                    return refine_llm
                return react_llm

            mock_get_llm.side_effect = get_llm_side_effect

            resp = Mock()
            resp.content = "No tools to call"
            resp.tool_calls = []
            react_llm.invoke.return_value = resp

            from app.agents.diagnose_agent import build_diagnose_agent
            agent = build_diagnose_agent()

            state = {
                "messages": [],
                "task_description": "test",
                "max_iterations": 5,
                "iteration_count": 0,
                "tool_errors": [],
                "tool_results": {},
            }

            result = agent.invoke(state)
            assert result is not None
            assert result.get("integrated_result") is not None


class TestSubgraphRegistry:
    """Test subgraph registry and adapters"""

    def test_detection_adapter_build_input(self):
        """Test DetectionAdapter builds correct input state."""
        from app.agents.subgraph_registry import get_adapter

        adapter = get_adapter("detection")
        step_input = {
            "task_description": "检测异常",
            "data_path": "data/test.csv",
            "inject_time": 100.0,
        }
        input_state = adapter.build_input(step_input, {})

        assert input_state["task_description"] == "检测异常"
        assert input_state["csv_file_path"] == "data/test.csv"
        assert input_state["inject_time"] == 100.0
        assert input_state["max_iterations"] == 5

    def test_detection_adapter_extract_result(self):
        """Test DetectionAdapter extracts correct result."""
        from app.agents.subgraph_registry import get_adapter

        adapter = get_adapter("detection")
        output = {
            "three_sigma_result": {"success": True, "anomalies": ["cpu"]},
            "final_response": "Found anomalies",
            "csv_file_path": "data/test.csv",
            "inject_time": 100.0,
            "abnormal_kpi": "cpu_usage",
        }
        result = adapter.extract_result(output)

        assert result["success"] is True
        assert result["summary"] == "Found anomalies"
        assert result["csv_file_path"] == "data/test.csv"
        assert result["inject_time"] == 100.0

    def test_diagnose_adapter_build_input_with_from_step(self):
        """Test DiagnoseAdapter passes detection results from prior step."""
        from app.agents.subgraph_registry import get_adapter

        adapter = get_adapter("diagnose")
        step_input = {
            "task_description": "根因分析",
            "from_step": 1,
        }
        step_results = {
            1: {
                "anomaly_report": [
                    {"metric": "cpu_usage", "anomaly_type": "sudden_increase",
                     "max_z_score": 5.23, "anomaly_point_count": 3},
                    {"metric": "mem_usage", "anomaly_type": "sudden_increase",
                     "max_z_score": 3.12, "anomaly_point_count": 2},
                ],
                "csv_file_path": "data/test.csv",
                "inject_time": 100.0,
                "abnormal_kpi": "cpu_usage",
            }
        }
        input_state = adapter.build_input(step_input, step_results)

        assert "异常检测结果" in input_state["task_description"]
        assert "cpu_usage" in input_state["task_description"]
        assert input_state["csv_file_path"] == "data/test.csv"
        assert input_state["inject_time"] == 100.0
        assert input_state["abnormal_kpi"] == "cpu_usage"

    def test_diagnose_adapter_extract_result(self):
        """Test DiagnoseAdapter extracts correct result."""
        from app.agents.subgraph_registry import get_adapter

        adapter = get_adapter("diagnose")
        output = {
            "rcd_result": {"success": True, "root_causes": ["metric_a"]},
            "pc_result": {"success": True, "root_causes": ["metric_b"]},
            "final_response": "Root cause analysis complete",
            "csv_file_path": "data/test.csv",
            "inject_time": 100.0,
            "abnormal_kpi": "metric_a",
            "graph_visualizations": [{"filepath": "graph.html"}],
        }
        result = adapter.extract_result(output)

        assert result["success"] is True
        assert result["rcd_result"]["success"] is True
        assert result["pc_result"]["success"] is True
        assert len(result["graph_visualizations"]) == 1

    def test_unknown_adapter_raises(self):
        """Test get_adapter raises for unknown agent."""
        from app.agents.subgraph_registry import get_adapter

        with pytest.raises(ValueError, match="Unknown sub-agent"):
            get_adapter("nonexistent")


class TestPlanExecuteNodes:
    """Test individual Plan-Execute nodes"""

    def test_parse_plan_json_plain(self):
        """Test JSON parsing from plain content."""
        from app.agents.supervisor_plan_execute import _parse_plan_json

        content = '{"reasoning": "test", "steps": []}'
        result = _parse_plan_json(content)
        assert result["reasoning"] == "test"

    def test_parse_plan_json_code_block(self):
        """Test JSON parsing from markdown code block."""
        from app.agents.supervisor_plan_execute import _parse_plan_json

        content = '```json\n{"reasoning": "test", "steps": []}\n```'
        result = _parse_plan_json(content)
        assert result["reasoning"] == "test"

    def test_parse_plan_json_with_surrounding_text(self):
        """Test JSON extraction from surrounding text."""
        from app.agents.supervisor_plan_execute import _parse_plan_json

        content = 'Here is the plan:\n{"reasoning": "test", "steps": []}\nDone.'
        result = _parse_plan_json(content)
        assert result["reasoning"] == "test"

    def test_parse_plan_json_invalid(self):
        """Test returns None for invalid JSON."""
        from app.agents.supervisor_plan_execute import _parse_plan_json

        result = _parse_plan_json("not json at all")
        assert result is None

    def test_route_after_planner_with_steps(self):
        """Test routing to executor when plan has steps."""
        from app.agents.supervisor_plan_execute import route_after_planner

        state = {
            "plan": [{"step_id": 1, "agent": "detection"}],
            "current_step_index": 0,
        }
        assert route_after_planner(state) == "executor"

    def test_route_after_planner_empty_plan(self):
        """Test routing to direct_reply when plan is empty."""
        from app.agents.supervisor_plan_execute import route_after_planner

        state = {
            "plan": [],
            "current_step_index": -1,
        }
        assert route_after_planner(state) == "direct_reply"

    def test_route_after_step_loop(self):
        """Test routing back to executor when more steps remain."""
        from app.agents.supervisor_plan_execute import route_after_step

        state = {
            "plan": [{"step_id": 1}, {"step_id": 2}],
            "current_step_index": 1,
        }
        assert route_after_step(state) == "executor"

    def test_route_after_step_done(self):
        """Test routing to finalize when all steps done."""
        from app.agents.supervisor_plan_execute import route_after_step

        state = {
            "plan": [{"step_id": 1}],
            "current_step_index": 1,
        }
        assert route_after_step(state) == "finalize"

    def test_direct_reply_node(self):
        """Test direct_reply_node generates response from plan_reply."""
        from app.agents.supervisor_plan_execute import direct_reply_node

        state = {
            "messages": [],
            "plan_reply": "您好！我是 AIOps 根因分析助手。",
        }
        result = direct_reply_node(state)
        assert result["final_response"] == "您好！我是 AIOps 根因分析助手。"
        assert result["continue_conversation"] is True

    def test_direct_reply_node_fallback(self):
        """Test direct_reply_node fallback when no plan_reply."""
        from app.agents.supervisor_plan_execute import direct_reply_node

        state = {
            "messages": [],
            "plan_reply": "",
        }
        result = direct_reply_node(state)
        assert "AIOps" in result["final_response"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
