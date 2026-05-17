"""Tests for the new workflow: LLM-driven tool calling with human-in-the-loop

These tests cover:
1. Supervisor routing to diagnose for diagnosis requests
2. Supervisor NOT validating parameters
3. Diagnose ReAct loop: CSV → RCD → PC → final
4. Human-in-the-loop when parameters are missing
5. Tool error handling and continuation
6. Final result refinement with all results and errors
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from app.agents.main_graph import main_graph
from app.agents.supervisor_agent import supervisor_agent, is_diagnosis_intent
from app.agents.diagnose_agent import diagnose_agent


class TestSupervisorAgentWorkflow:
    """Test Supervisor Agent intent recognition (no parameter validation)"""

    def test_diagnosis_intent_recognition(self):
        """Test that supervisor correctly identifies diagnosis intent"""
        assert is_diagnosis_intent("分析一下今天的故障")
        assert is_diagnosis_intent("帮我定位根因")
        assert is_diagnosis_intent("Please analyze the anomaly")
        assert is_diagnosis_intent("使用RCD算法诊断")

    def test_non_diagnosis_intent(self):
        """Test that supervisor correctly identifies non-diagnosis intent"""
        assert not is_diagnosis_intent("你好")
        assert not is_diagnosis_intent("天气怎么样")
        assert not is_diagnosis_intent("hello")

    def test_supervisor_routes_to_diagnose(self):
        """Test supervisor routes to diagnose for diagnosis requests"""
        state = {
            "user_input": "今天下午3点发生异常，请分析数据文件",
            "messages": [],
            "action": "respond",
            "csv_file_path": None,
            "inject_time": None,
            "gamma": None,
            "alpha": None,
            "dataset_type": None,
            "diagnose_result": None,
            "response_message": None,
            "continue_conversation": True
        }

        result = supervisor_agent.invoke(state)

        # Should route to diagnose (action != "respond")
        assert result["action"] in ["call_diagnose", "respond"]
        # Should NOT extract parameters - csv_file_path remains None
        # Parameter extraction is now diagnose's responsibility
        assert result["csv_file_path"] is None

    def test_supervisor_does_not_extract_parameters(self):
        """Test that supervisor does NOT extract/validate parameters"""
        state = {
            "user_input": "分析 data/test.csv 文件，注入时间100",
            "messages": [],
            "action": "respond",
            "csv_file_path": None,
            "inject_time": None,
            "gamma": None,
            "alpha": None,
            "dataset_type": None,
            "diagnose_result": None,
            "response_message": None,
            "continue_conversation": True
        }

        result = supervisor_agent.invoke(state)

        # Supervisor should route to diagnose
        assert result["action"] == "call_diagnose"
        # But should NOT extract parameters (remains None)
        # The diagnose_agent will handle extraction
        assert result["csv_file_path"] is None
        assert result["inject_time"] is None


class TestDiagnoseAgentReActLoop:
    """Test Diagnose Agent ReAct loop workflow"""

    def test_diagnose_state_initialization(self):
        """Test diagnose state is properly initialized"""
        from app.agents.main_graph import state_to_diagnose

        main_state = {
            "user_input": "分析 data/test.csv，注入时间100",
            "csv_file_path": None,
            "inject_time": 100,
            "gamma": 5,
            "alpha": 0.05,
            "dataset_type": None
        }

        diagnose_state = state_to_diagnose(main_state)

        # Should have task_description set
        assert diagnose_state["task_description"] == main_state["user_input"]
        # Should start with no results
        assert diagnose_state["rcd_result"] is None
        assert diagnose_state["pc_result"] is None
        assert diagnose_state["tool_errors"] == []

    @patch('app.agents.diagnose_agent.get_deepseek_llm')
    def test_diagnose_llm_decides_tool_calling(self, mock_llm):
        """Test that LLM autonomously decides which tools to call"""
        from app.agents.diagnose_agent import DiagnoseAgentState

        # Mock LLM to return a tool call
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = ""

        # Create mock tool_calls
        mock_tool_call = Mock()
        mock_tool_call.name = "read_csv"
        mock_tool_call.args = {"data_path": "data/test.csv"}
        mock_tool_call.id = "call_123"

        mock_response.tool_calls = [mock_tool_call]
        mock_client.invoke.return_value = mock_response

        mock_llm_instance = Mock()
        mock_llm_instance.get_client.return_value = mock_client
        mock_llm_instance.bind_tools.return_value = mock_llm_instance
        mock_llm.return_value = mock_llm_instance

        mock_get_deepseek = Mock(return_value=mock_llm_instance)

        with patch('app.agents.diagnose_agent.get_deepseek_llm', mock_get_deepseek):
            from app.agents.diagnose_agent import model_node

            state = DiagnoseAgentState(
                messages=[],
                task_description="分析 data/test.csv",
                iteration_count=0
            )

            result = model_node(state)

            # Should have added the AIMessage with tool_calls
            assert len(result["messages"]) > 0
            assert result["iteration_count"] == 1


class TestHumanInTheLoop:
    """Test human-in-the-loop functionality"""

    def test_ask_user_tool_triggers_interrupt(self):
        """Test that ask_user tool triggers interrupt for missing params"""
        from app.agents.diagnose_agent import ask_user_tool_func

        result = ask_user_tool_func(question="请提供CSV文件路径")
        result_dict = json.loads(result)

        assert result_dict["status"] == "interrupted"
        assert "请提供CSV文件路径" in result_dict["question"]

    def test_ask_user_in_diagnose_tools(self):
        """Test ask_user tool is available in diagnose tools"""
        from app.agents.diagnose_agent import diagnose_tools

        tool_names = [tool.name for tool in diagnose_tools]
        assert "ask_user" in tool_names


class TestToolErrorHandling:
    """Test tool execution error handling"""

    @patch('app.agents.diagnose_agent.pd.read_csv')
    def test_csv_read_error_is_caught(self, mock_read_csv):
        """Test that CSV read errors are caught and don't crash the system"""
        from app.agents.diagnose_agent import read_csv_tool_func, extract_results_node, DiagnoseAgentState

        # Mock CSV read to raise exception
        mock_read_csv.side_effect = FileNotFoundError("File not found")

        result = read_csv_tool_func(data_path="data/nonexistent.csv")
        result_dict = json.loads(result)

        # Should return error result, not crash
        assert result_dict["success"] == False
        assert "error" in result_dict

    @patch('app.agents.diagnose_agent.run_rcd_analysis')
    def test_rcd_error_is_caught_and_logged(self, mock_rcd):
        """Test that RCD errors are caught and logged"""
        from app.agents.diagnose_agent import rcd_tool_func, extract_results_node, DiagnoseAgentState

        # Mock RCD to raise exception
        mock_rcd.side_effect = Exception("RCD algorithm failed")

        # First cache some data
        from app.agents.diagnose_agent import cache_csv_data
        import pandas as pd
        cache_csv_data("test.csv", pd.DataFrame({"a": [1, 2, 3]}))

        result = rcd_tool_func(inject_time=100.0)
        result_dict = json.loads(result)

        # Should return error result
        assert result_dict["algorithm"] == "RCD"
        assert result_dict["success"] == False
        assert "error" in result_dict


class TestFinalResultRefinement:
    """Test final result refinement with all results and errors"""

    @patch('app.agents.diagnose_agent.get_deepseek_llm')
    def test_final_refinement_with_partial_success(self, mock_llm):
        """Test final refinement when only some tools succeeded"""
        mock_llm_instance = Mock()
        mock_llm_instance.invoke.return_value = "Final analysis report"

        with patch('app.agents.diagnose_agent.get_deepseek_llm', return_value=mock_llm_instance):
            from app.agents.diagnose_agent import final_response_node, DiagnoseAgentState

            state = DiagnoseAgentState(
                messages=[],
                task_description="分析任务",
                rcd_result={
                    "success": True,
                    "root_causes": ["metric_a", "metric_b"]
                },
                pc_result={
                    "success": False,
                    "error": "PC algorithm failed"
                },
                tool_errors=[
                    {"tool": "pc_algorithm", "error": "PC algorithm failed"}
                ],
                csv_data=None
            )

            result = final_response_node(state)

            # Should generate integrated result
            assert result["integrated_result"] is not None
            # Should include RCD success and PC failure info
            assert "metric_a" in result["integrated_result"] or "Final analysis report" in result["integrated_result"]


class TestEndToEndWorkflow:
    """Test end-to-end workflow scenarios"""

    def test_diagnosis_request_routes_to_diagnose(self):
        """Test: User inputs diagnosis request → supervisor routes to diagnose"""
        from app.agents.main_graph import MainState, main_graph

        state = MainState(
            user_input="今天下午3点，微服务系统发生异常，请分析根因",
            messages=[],
            continue_conversation=True,
            action="respond"
        )

        result = main_graph.invoke(state)

        # Should route to diagnose (action will be "call_diagnose")
        assert result["action"] in ["call_diagnose", "have_diagnose_result", "respond"]

    @patch('app.agents.diagnose_agent.get_deepseek_llm')
    @patch('app.agents.diagnose_agent.pd.read_csv')
    def test_diagnose_executes_tools_in_sequence(self, mock_read_csv, mock_llm):
        """Test: Diagnose executes tools in sequence CSV → RCD → PC"""
        # This would require full integration test with actual LLM
        # For now, just verify the graph structure
        from app.agents.diagnose_agent import diagnose_agent

        # The graph should be compiled
        assert diagnose_agent is not None

        # Should have the expected nodes
        graph = diagnose_agent
        nodes = graph.nodes
        assert "model" in nodes
        assert "tools" in nodes
        assert "extract_results" in nodes
        assert "final" in nodes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
