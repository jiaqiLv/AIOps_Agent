"""Generic ReAct Loop Nodes

This module provides generic node implementations for configuration-driven
ReAct loops where the LLM autonomously decides which tools to call.

Nodes:
- model_node: Invokes LLM with bound tools, returns AIMessage with tool_calls
- tool_node: Executes tool calls via LangChain ToolNode
- extract_results_node: Parses ToolMessages, updates state, detects interrupts
- final_response_node: Synthesizes final report using refine prompt

Routing:
- route_after_model: Routes based on tool_calls presence and iteration count
- route_after_extract: Routes based on interrupt status and completion
"""

import json
import uuid
from typing import Dict, Any, List, Optional, Callable
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from langgraph.graph import END

from app.models.react_agent_state import ReactAgentState
from app.utils.logger import get_logger
from app.utils.prompt_loader import load_prompt
from app.utils.llm_logger import log_llm_conversation

logger = get_logger(__name__)


# ==================== Message Compression ====================

def compress_messages(messages: List, max_length: int = 2000) -> List:
    """Compress long ToolMessages to reduce token usage.

    Keeps essential keys and truncates large lists.
    """
    compressed = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and len(msg.content) > max_length:
            try:
                content = json.loads(msg.content)
                # Keep only essential keys
                essential_keys = {
                    "success", "status", "error", "root_causes", "edges",
                    "shape", "columns", "data_path", "message", "algorithm"
                }
                compressed_content = {
                    k: v for k, v in content.items()
                    if k in essential_keys or (isinstance(v, (str, int, float, bool)) and len(str(v)) < 200)
                }
                # Truncate large lists
                for key in ["root_causes", "edges", "columns"]:
                    if key in compressed_content and isinstance(compressed_content[key], list):
                        compressed_content[key] = compressed_content[key][:20]
                compressed_msg = ToolMessage(
                    content=json.dumps(compressed_content, ensure_ascii=False),
                    tool_call_id=msg.tool_call_id,
                    name=msg.name
                )
                compressed.append(compressed_msg)
            except (json.JSONDecodeError, TypeError):
                # If compression fails, keep original
                compressed.append(msg)
        else:
            compressed.append(msg)
    return compressed


# ==================== Node Factory Functions ====================

def create_model_node(llm, system_prompt: str) -> Callable:
    """Create a model node that invokes the LLM with bound tools.

    Args:
        llm: LLM instance with tools bound via bind_tools()
        system_prompt: System prompt to guide the LLM

    Returns:
        Node function for StateGraph
    """
    def model_node(state: ReactAgentState) -> ReactAgentState:
        """Invoke LLM to decide next action (tool call or final response)."""
        iteration = state.get("iteration_count", 0) + 1
        state["iteration_count"] = iteration

        # Prepare messages with system prompt
        messages = list(state.get("messages", []))

        # If messages is empty but we have task_description, create initial HumanMessage
        if not messages:
            task_description = state.get("task_description", "")
            if task_description:
                messages = [HumanMessage(content=task_description)]
                logger.info(f"REACT: Initial task: {task_description[:100]}...")

        # Add system prompt as first message if not present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages

        # For first iteration, add explicit instruction to call tools
        if iteration == 1:
            instruction = HumanMessage(content="\n\n请立即调用 csv_reader_tool 开始分析数据。")
            messages = messages + [instruction]

        # Compress long messages
        messages = compress_messages(messages)

        logger.info(f"REACT: Iteration {iteration}/{state.get('max_iterations', 10)}, invoking LLM")

        try:
            response = llm.invoke(messages)

            # Ensure response is an AIMessage
            if isinstance(response, str):
                response = AIMessage(content=response)
                logger.warning("REACT: LLM returned string, converted to AIMessage")
            elif not isinstance(response, AIMessage):
                logger.warning(f"REACT: LLM returned unexpected type: {type(response)}")
                response = AIMessage(content=str(response))

            state["messages"] = messages + [response]

            # Log for debugging
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_names = [tc.get('name', 'unknown') for tc in response.tool_calls]
                logger.info(f"REACT: LLM requested tools: {tool_names}")
            else:
                logger.info("REACT: LLM did not request tools (proceeding to final)")

            log_llm_conversation(
                agent_name="react_model",
                iteration=iteration,
                input_messages=messages,
                response=response,
                metadata={
                    "has_tool_calls": bool(hasattr(response, 'tool_calls') and response.tool_calls),
                    "tool_count": len(response.tool_calls) if hasattr(response, 'tool_calls') else 0
                }
            )

        except Exception as e:
            logger.error(f"REACT: LLM invocation failed: {e}")
            # Add error message and continue
            error_msg = AIMessage(content=f"LLM 调用失败: {str(e)}")
            state["messages"] = messages + [error_msg]

        return state

    return model_node


def extract_results_node(state: ReactAgentState) -> ReactAgentState:
    """Extract results from ToolMessages and update state.

    This node:
    1. Parses ToolMessages to extract structured results
    2. Updates tool_results dict with parsed data
    3. Detects ask_user interrupt requests
    4. Tracks errors for final reporting
    5. Updates legacy fields (csv_file_path, rcd_result, etc.) for compatibility
    """
    messages = state.get("messages", [])

    # Initialize tool_results if not present
    if "tool_results" not in state:
        state["tool_results"] = {}

    # Initialize tool_errors if not present
    if "tool_errors" not in state:
        state["tool_errors"] = []

    # Find new ToolMessages (since last model call)
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

    for tool_msg in tool_messages:
        tool_name = tool_msg.name
        tool_call_id = tool_msg.tool_call_id

        logger.debug(f"REACT: Extracting results from {tool_name} (call_id: {tool_call_id})")

        try:
            result = json.loads(tool_msg.content)

            # Store in generic tool_results
            state["tool_results"][tool_name] = result

            # Update legacy fields for compatibility
            if tool_name == "csv_reader_tool" or tool_name == "read_csv":
                if result.get("success"):
                    state["csv_file_path"] = result.get("data_path")
                    state["csv_headers"] = result.get("columns")
                    logger.info(f"REACT: CSV loaded from {result.get('data_path')}")
                else:
                    state["tool_errors"].append({
                        "tool": tool_name,
                        "error": result.get("error", "Unknown error"),
                        "iteration": state.get("iteration_count", 0)
                    })

            elif tool_name == "rcd_tool" or tool_name == "rcd_algorithm":
                state["rcd_result"] = result
                if result.get("success"):
                    root_causes = result.get("root_causes", [])
                    logger.info(f"REACT: RCD completed with {len(root_causes)} root causes")
                else:
                    state["tool_errors"].append({
                        "tool": tool_name,
                        "error": result.get("error", "Unknown error"),
                        "iteration": state.get("iteration_count", 0)
                    })

            elif tool_name == "pc_tool" or tool_name == "pc_algorithm":
                state["pc_result"] = result
                if result.get("success"):
                    root_causes = result.get("root_causes", [])
                    edges = result.get("edges", [])
                    logger.info(f"REACT: PC completed with {len(root_causes)} root causes, {len(edges)} edges")
                else:
                    state["tool_errors"].append({
                        "tool": tool_name,
                        "error": result.get("error", "Unknown error"),
                        "iteration": state.get("iteration_count", 0)
                    })

            elif tool_name == "ask_user":
                # Handle interrupt request
                if result.get("requires_user_input"):
                    state["interrupted"] = True
                    state["interrupt_data"] = {
                        "question": result.get("question"),
                        "tool": result.get("tool"),
                        "missing_params": result.get("missing_params", [])
                    }
                    logger.info(f"REACT: Interrupt requested - question: {result.get('question')}")

            elif tool_name == "graph_visualization_tool":
                state["tool_results"][tool_name] = result
                if result.get("success"):
                    filepath = result.get("filepath")
                    logger.info(f"REACT: Graph visualization generated: {filepath}")
                    # Store for final report
                    if not state.get("graph_visualizations"):
                        state["graph_visualizations"] = []
                    state["graph_visualizations"].append(result)
                else:
                    state["tool_errors"].append({
                        "tool": tool_name,
                        "error": result.get("error", "Unknown error"),
                        "iteration": state.get("iteration_count", 0)
                    })

        except json.JSONDecodeError:
            logger.warning(f"REACT: Failed to parse tool result as JSON: {tool_msg.content[:200]}")
            state["tool_errors"].append({
                "tool": tool_name,
                "error": f"Failed to parse result: {tool_msg.content[:200]}",
                "iteration": state.get("iteration_count", 0)
            })

    return state


def create_final_response_node(refine_prompt_path: str = "app/prompts/diagnose_refine.md",
                                llm=None) -> Callable:
    """Create a final response node that synthesizes all results.

    Args:
        refine_prompt_path: Path to the refine prompt template
        llm: LLM instance for final synthesis (uses default if None)

    Returns:
        Node function for StateGraph
    """
    def final_response_node(state: ReactAgentState) -> ReactAgentState:
        """Generate final response from all tool results."""
        logger.info("REACT: Generating final response")

        # Build result summary
        parts = []

        task_description = state.get("task_description", "")
        if task_description:
            parts.append(f"## 任务描述\n{task_description}")

        # Add analysis parameters
        inject_time = state.get("inject_time")
        abnormal_kpi = state.get("abnormal_kpi")
        if inject_time or abnormal_kpi:
            parts.append("\n## 分析参数")
            if inject_time:
                from datetime import datetime, timezone, timedelta
                tz = timezone(timedelta(hours=8))
                dt_str = datetime.fromtimestamp(inject_time, tz=tz).strftime("%Y-%m-%d %H:%M:%S")
                parts.append(f"- 故障注入时间: {dt_str} (Unix: {inject_time})")
            if abnormal_kpi:
                parts.append(f"- 异常指标: {abnormal_kpi}")

        # Add CSV info
        csv_path = state.get("csv_file_path")
        csv_headers = state.get("csv_headers")
        if csv_path and csv_headers:
            parts.append(f"\n## CSV 数据\n- 文件: {csv_path}\n- 列数: {len(csv_headers)}")

        # Add RCD results
        rcd_result = state.get("rcd_result")
        if rcd_result:
            parts.append("\n## IAF-RCL 算法结果")
            if rcd_result.get("success"):
                rc = rcd_result.get("root_causes", [])
                parts.append(f"- 状态: 成功\n- 根因数量: {len(rc)}")
                if rc:
                    parts.append(f"- 根因列表 (前20): {rc[:20]}")
            else:
                parts.append(f"- 状态: 失败\n- 错误: {rcd_result.get('error', 'Unknown')}")

        # Add PC results
        pc_result = state.get("pc_result")
        if pc_result:
            parts.append("\n## KE-FPC 算法结果")
            if pc_result.get("success"):
                rc = pc_result.get("root_causes", [])
                edges = pc_result.get("edges", [])
                parts.append(f"- 状态: 成功\n- 根因数量: {len(rc)}\n- 因果边: {len(edges)}")
                if rc:
                    parts.append(f"- 根因列表 (前20): {rc[:20]}")
                if edges:
                    parts.append(f"- 因果边 (前30): {edges[:30]}")
            else:
                parts.append(f"- 状态: 失败\n- 错误: {pc_result.get('error', 'Unknown')}")

        # Add graph visualizations
        graph_visualizations = state.get("graph_visualizations", [])
        if graph_visualizations:
            parts.append("\n## 故障传播图可视化")
            for viz in graph_visualizations:
                filepath = viz.get("filepath")
                fmt = viz.get("format", "html")
                parts.append(f"- {fmt.upper()} 格式图: {filepath}")
                if viz.get("abnormal_kpi"):
                    parts.append(f"  异常指标: {viz['abnormal_kpi']}")

        # Add tool errors
        tool_errors = state.get("tool_errors", [])
        if tool_errors:
            parts.append("\n## 执行错误")
            for err in tool_errors:
                parts.append(f"- {err.get('tool', 'unknown')}: {err.get('error', 'unknown')}")

        result_str = "\n".join(parts)

        # Load refine prompt
        try:
            refine_prompt = load_prompt(refine_prompt_path)
            prompt = refine_prompt.replace("{{RESULT_STR}}", result_str)
        except Exception:
            if rcd_result or pc_result:
                prompt = f"你是一个 AIOps 根因分析专家。基于以下分析结果生成报告：\n\n{result_str}\n\n请生成包含根因指标列表、故障传播路径、故障类型判断、结论与建议的结构化报告。"
            else:
                prompt = f"未能成功执行任何根因分析算法。\n\n任务: {task_description}\n\n请说明问题并给出建议。"

        # Generate final response with LLM
        if llm is None:
            from app.config.model_config import get_deepseek_llm
            llm_instance = get_deepseek_llm(temperature=0.3)
        else:
            llm_instance = llm

        try:
            final_result = llm_instance.invoke(prompt)
            final_content = final_result if isinstance(final_result, str) else str(final_result)

            log_llm_conversation(
                agent_name="react_final",
                iteration=1,
                input_messages=[HumanMessage(content=prompt)],
                response=AIMessage(content=final_content),
                metadata={
                    "rcd_executed": bool(rcd_result),
                    "pc_executed": bool(pc_result),
                    "tool_errors_count": len(tool_errors),
                    "type": "final_refinement"
                }
            )

            state["final_response"] = final_content
            state["integrated_result"] = final_content  # Legacy compatibility

        except Exception as e:
            logger.error(f"REACT: Final LLM call failed: {e}")
            state["final_response"] = result_str + "\n\n注: LLM报告生成失败，以上为原始执行结果。"
            state["integrated_result"] = state["final_response"]

        graph_viz = None
        graph_visualizations = state.get("graph_visualizations") or []
        if graph_visualizations:
            graph_viz = graph_visualizations[-1]

        from app.utils.topology_chat import build_final_report_message

        state["messages"] = state.get("messages", []) + [
            build_final_report_message(state["final_response"], graph_viz)
        ]

        logger.info("REACT: Final response generated")
        return state

    return final_response_node


# ==================== Routing Functions ====================

def route_after_model(state: Dict) -> str:
    """Route after model_node based on LLM response.

    Returns:
        "tools" if LLM made tool_calls
        "final" if LLM didn't make tool_calls
        "final" if max_iterations reached
    """
    messages = state.get("messages", [])
    if not messages:
        return "final"

    last_message = messages[-1]

    # Check for tool_calls
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        # Check iteration limit
        iteration = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 10)
        if iteration >= max_iterations:
            logger.info(f"REACT: Max iterations ({max_iterations}) reached, proceeding to final")
            return "final"
        return "tools"

    return "final"


def route_after_extract(state: Dict) -> str:
    """Route after extract_results_node based on interrupt status.

    Returns:
        "interrupt" if ask_user triggered interrupt
        "model" to continue the ReAct loop
        "final" if no more tools to call
    """
    # Check for interrupt
    if state.get("interrupted"):
        logger.info("REACT: Interrupt requested, triggering LangGraph interrupt()")
        return "interrupt"

    # Check if we have meaningful results
    tool_results = state.get("tool_results", {})
    messages = state.get("messages", [])

    # Find the last AIMessage to see if it made tool_calls
    last_ai_message = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            last_ai_message = msg
            break

    # If last AI message had tool_calls, we should continue the loop
    if last_ai_message and hasattr(last_ai_message, 'tool_calls') and last_ai_message.tool_calls:
        # Check if we haven't exceeded max iterations
        iteration = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 10)
        if iteration < max_iterations:
            return "model"

    # Otherwise, proceed to final
    return "final"
