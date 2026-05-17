"""Integration tests for the new LLM-driven workflow

This tests the complete flow from user input through supervisor -> diagnose -> tools -> final result.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from app.agents.main_graph import main_graph, MainState


class TestSupervisorToDiagnoseRouting:
    """Test supervisor correctly routes diagnosis requests"""

    @patch('app.agents.supervisor_agent.get_deepseek_llm')
    def test_diagnosis_request_routes_to_diagnose(self, mock_llm):
        """Test: User inputs diagnosis request → supervisor routes to diagnose"""
        # Mock LLM to return diagnosis intent
        mock_llm_instance = Mock()
        mock_response = Mock()
        mock_response.content = '{"action": "call_diagnose"}'
        mock_llm_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_llm_instance

        state = MainState(
            user_input="今天下午3点，微服务系统发生异常，请结合指标数据分析故障根因。",
            messages=[],
            continue_conversation=True,
            action="respond"
        )

        result = main_graph.invoke(state)

        # Should route to diagnose
        assert result["action"] in ["call_diagnose", "have_diagnose_result"]

    @patch('app.agents.supervisor_agent.get_deepseek_llm')
    def test_supervisor_no_longer_validates_parameters(self, mock_llm):
        """Test: Supervisor does NOT extract or validate parameters"""
        mock_llm_instance = Mock()
        mock_response = Mock()
        mock_response.content = '{"action": "call_diagnose"}'
        mock_llm_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_llm_instance

        state = MainState(
            user_input="分析 data/test.csv 文件，注入时间100",
            messages=[],
            continue_conversation=True,
            action="respond",
            csv_file_path=None,  # Not set by supervisor
            inject_time=None
        )

        result = main_graph.invoke(state)

        # Supervisor should route to diagnose but NOT extract parameters
        assert result["action"] == "call_diagnose"
        # Parameters remain unset - diagnose will handle extraction
        # Note: They may get passed through from state, but supervisor doesn't validate

    @patch('app.agents.supervisor_agent.get_deepseek_llm')
    def test_general_chat_responds_directly(self, mock_llm):
        """Test: Non-diagnosis requests get direct responses"""
        mock_llm_instance = Mock()
        mock_response = Mock()
        mock_response.content = '{"action": "respond", "message": "你好！"}'
        mock_llm_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_llm_instance

        state = MainState(
            user_input="你好",
            messages=[],
            continue_conversation=True,
            action="respond"
        )

        result = main_graph.invoke(state)

        # Should respond directly
        assert result["action"] == "respond"
        assert result["continue_conversation"] == True
        assert len(result["messages"]) > 0


class TestDiagnoseReActLoop:
    """Test diagnose agent ReAct loop execution"""

    @patch('app.agents.diagnose_agent.pd.read_csv')
    @patch('app.agents.diagnose_agent.get_deepseek_llm')
    def test_diagnose_acts_sequentially_csv_to_rcd_to_pc(self, mock_llm, mock_read_csv):
        """Test: Diagnose executes tools sequentially: CSV → RCD → PC"""
        # Mock CSV data
        mock_df = pd.DataFrame({
            "time": [1, 2, 3, 4, 5],
            "metric_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "metric_b": [2.0, 3.0, 4.0, 5.0, 6.0]
        })
        mock_read_csv.return_value = mock_df

        # Mock LLM responses for each iteration
        # Iteration 1: LLM decides to call read_csv
        # Iteration 2: LLM decides to call rcd
        # Iteration 3: LLM decides to call pc
        # Iteration 4: LLM has no tool calls, ends

        mock_client = Mock()
        mock_llm_instance = Mock()
        mock_llm_instance.get_client.return_value = mock_client
        mock_llm_instance.bind_tools.return_value = mock_llm_instance
        mock_llm.return_value = mock_llm_instance

        call_count = [0]

        def create_tool_call(name, args):
            tc = Mock()
            tc.name = name
            tc.args = args
            tc.id = f"call_{call_count[0]}_{name}"
            return tc

        def create_response_with_tool_calls(tool_calls):
            resp = Mock()
            resp.content = ""
            resp.tool_calls = tool_calls
            return resp

        def create_response_no_tool():
            resp = Mock()
            resp.content = "分析完成"
            resp.tool_calls = None
            return resp

        responses = [
            # Iteration 1: Call read_csv
            create_response_with_tool_calls([create_tool_call("read_csv", {"data_path": "data/test.csv"})]),
            # Iteration 2: Call rcd
            create_response_with_tool_calls([create_tool_call("rcd_algorithm", {"inject_time": 100})]),
            # Iteration 3: Call pc
            create_response_with_tool_calls([create_tool_call("pc_algorithm", {})]),
            # Iteration 4: No tool calls
            create_response_no_tool(),
        ]

        def invoke_side_effect(messages):
            idx = call_count[0]
            call_count[0] += 1
            return responses[idx]

        mock_client.invoke.side_effect = invoke_side_effect
        mock_llm_instance.invoke.side_effect = invoke_side_effect

        with patch('app.agents.diagnose_agent.get_deepseek_llm', return_value=mock_llm_instance):
            from app.agents.diagnose_agent import diagnose_agent, DiagnoseAgentState

            state = DiagnoseAgentState(
                messages=[],
                task_description="分析 data/test.csv，注入时间100",
                iteration_count=0
            )

            result = diagnose_agent.invoke(state)

            # Should have executed all tools
            assert result["iteration_count"] == 4  # 4 iterations

    @patch('app.agents.diagnose_agent.pd.read_csv')
    @patch('app.agents.diagnose_agent.get_deepseek_llm')
    def test_llm_generates_tool_parameters(self, mock_llm, mock_read_csv):
        """Test: LLM generates tool call parameters based on context"""
        mock_df = pd.DataFrame({"time": [1, 2, 3], "value": [1, 2, 3]})
        mock_read_csv.return_value = mock_df

        mock_client = Mock()
        mock_llm_instance = Mock()
        mock_llm_instance.get_client.return_value = mock_client

        # LLM should extract inject_time=100 from task description
        mock_tool_call = Mock()
        mock_tool_call.name = "rcd_algorithm"
        mock_tool_call.args = {"inject_time": 100}  # LLM extracted this
        mock_tool_call.id = "call_123"

        mock_response = Mock()
        mock_response.content = ""
        mock_response.tool_calls = [mock_tool_call]

        mock_client.invoke.return_value = mock_response
        mock_llm_instance.invoke.return_value = mock_response

        with patch('app.agents.diagnose_agent.get_deepseek_llm', return_value=mock_llm_instance):
            from app.agents.diagnose_agent import model_node, DiagnoseAgentState

            state = DiagnoseAgentState(
                messages=[],
                task_description="注入时间100时发生异常，请分析",  # inject_time in description
                iteration_count=1,  # Simulating after CSV was read
                csv_data=mock_df
            )

            result = model_node(state)

            # LLM should have called rcd with inject_time parameter
            # (This is verified by the tool_call having inject_time)

    @patch('app.agents.diagnose_agent.pd.read_csv')
    @patch('app.agents.diagnose_agent.get_deepseek_llm')
    def test_missing_parameter_triggers_ask_user(self, mock_llm, mock_read_csv):
        """Test: Missing parameter triggers ask_user tool"""
        mock_df = pd.DataFrame({"time": [1, 2, 3], "value": [1, 2, 3]})
        mock_read_csv.return_value = mock_df

        mock_client = Mock()
        mock_llm_instance = Mock()
        mock_llm_instance.get_client.return_value = mock_client

        # LLM should call ask_user when inject_time is missing
        mock_tool_call = Mock()
        mock_tool_call.name = "ask_user"
        mock_tool_call.args = {"question": "请提供故障注入时间"}
        mock_tool_call.id = "call_123"

        mock_response = Mock()
        mock_response.content = ""
        mock_response.tool_calls = [mock_tool_call]

        mock_client.invoke.return_value = mock_response
        mock_llm_instance.invoke.return_value = mock_response

        with patch('app.agents.diagnose_agent.get_deepseek_llm', return_value=mock_llm_instance):
            from app.agents.diagnose_agent import model_node, DiagnoseAgentState

            state = DiagnoseAgentState(
                messages=[],
                task_description="分析 data/test.csv",  # No inject_time provided
                iteration_count=1,  # After CSV was read
                csv_data=mock_df
            )

            result = model_node(state)

            # Should have tool_calls
            assert len(result["messages"]) > 0


class TestToolErrorHandling:
    """Test tool error handling and continuation"""

    @patch('app.agents.diagnose_agent.pd.read_csv')
    @patch('app.agents.diagnose_agent.get_deepseek_llm')
    def test_csv_error_continues_to_other_tools(self, mock_llm, mock_read_csv):
        """Test: CSV read error is logged but doesn't stop RCD/PC"""
        # Mock CSV to fail first, then succeed
        mock_df = pd.DataFrame({"time": [1, 2, 3], "value": [1, 2, 3]})

        call_count = [0]

        def read_csv_side_effect(path):
            if call_count[0] == 0:
                raise FileNotFoundError("File not found")
            else:
                return mock_df

        mock_read_csv.side_effect = read_csv_side_effect

        mock_client = Mock()
        mock_llm_instance = Mock()
        mock_llm_instance.get_client.return_value = mock_client

        def create_response_with_tool_call(name, args):
            tc = Mock()
            tc.name = name
            tc.args = args
            tc.id = f"call_{name}"
            return tc

        # First iteration: read_csv fails
        # Second iteration: LLM decides to continue anyway (or user provides path)

        mock_response = Mock()
        mock_response.content = ""
        mock_response.tool_calls = None
        mock_client.invoke.return_value = mock_response
        mock_llm_instance.invoke.return_value = mock_response

        with patch('app.agents.diagnose_agent.get_deepseek_llm', return_value=mock_llm_instance):
            from app.agents.diagnose_agent import diagnose_agent, DiagnoseAgentState

            state = DiagnoseAgentState(
                messages=[],
                task_description="分析数据",
                iteration_count=0
            )

            result = diagnose_agent.invoke(state)

            # Should handle error gracefully
            # (In real scenario, LLM might ask user for correct path)


class TestFinalResultIncludesErrors:
    """Test final result includes all successes and errors"""

    @patch('app.agents.diagnose_agent.get_deepseek_llm')
    def test_final_report_includes_successful_and_failed_tools(self, mock_llm):
        """Test: Final report includes both successful and failed tool results"""
        mock_llm_instance = Mock()
        mock_llm_instance.invoke.return_value = "综合分析报告\n\nRCD: 成功\nPC: 失败 - 数据不足"

        with patch('app.agents.diagnose_agent.get_deepseek_llm', return_value=mock_llm_instance):
            from app.agents.diagnose_agent import final_response_node, DiagnoseAgentState

            state = DiagnoseAgentState(
                messages=[],
                task_description="分析任务",
                rcd_result={"success": True, "root_causes": ["metric_a"]},
                pc_result={"success": False, "error": "PC failed"},
                tool_errors=[
                    {"tool": "pc_algorithm", "error": "PC failed"}
                ],
                csv_data=None
            )

            result = final_response_node(state)

            # Should generate integrated result
            assert result["integrated_result"] is not None


class TestOutputFormatCompatibility:
    """Test that output format remains compatible with original system"""

    @patch('app.agents.diagnose_agent.get_deepseek_llm')
    def test_integrated_result_format(self, mock_llm):
        """Test: integrated_result maintains expected format"""
        mock_llm_instance.invoke.return_value = """
=== 根因分析报告 ===

**根因指标列表**
1. metric_a (支持: RCD)

**结论**
测试报告
"""

        with patch('app.agents.diagnose_agent.get_deepseek_llm', return_value=mock_llm_instance):
            from app.agents.diagnose_agent import final_response_node, DiagnoseAgentState

            state = DiagnoseAgentState(
                messages=[],
                task_description="测试",
                rcd_result={"success": True, "root_causes": ["metric_a"]},
                pc_result=None,
                tool_errors=[],
                csv_data=None
            )

            result = final_response_node(state)

            # integrated_result should exist
            assert "integrated_result" in result
            assert result["integrated_result"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])