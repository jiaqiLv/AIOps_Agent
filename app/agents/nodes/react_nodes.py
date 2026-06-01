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
from typing import Dict, Any, List, Callable
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from langgraph.graph import END

from app.utils.logger import get_logger
from app.utils.llm_logger import get_trace_logger

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

def create_model_node(llm, system_prompt: str, first_iteration_instruction: str = None) -> Callable:
    """Create a model node that invokes the LLM with bound tools.

    Args:
        llm: LLM instance with tools bound via bind_tools()
        system_prompt: System prompt to guide the LLM
        first_iteration_instruction: Optional instruction appended on the first iteration.
            If None, no extra instruction is added (supervisor decides autonomously).

    Returns:
        Node function for StateGraph
    """
    def model_node(state: Dict) -> Dict:
        """Invoke LLM to decide next action (tool call or final response)."""
        iteration = state.get("iteration_count", 0) + 1
        state["iteration_count"] = iteration

        # Prepare messages with system prompt
        messages = list(state.get("messages", []))
        # Track only NEW messages to return as delta (avoids add_messages duplication)
        new_messages = []

        # If messages is empty but we have task_description, create initial HumanMessage
        if not messages:
            task_description = state.get("task_description", "") or state.get("user_input", "")
            if task_description:
                human_msg = HumanMessage(content=task_description)
                messages = [human_msg]
                new_messages.append(human_msg)
                logger.info(f"REACT: Initial task: {task_description[:100]}...")

        # Add system prompt as first message if not present
        if not messages or not isinstance(messages[0], SystemMessage):
            sys_msg = SystemMessage(content=system_prompt)
            messages = [sys_msg] + messages
            new_messages.insert(0, sys_msg)

        # For first iteration, add explicit instruction if provided
        if iteration == 1 and first_iteration_instruction:
            instruction = HumanMessage(content=first_iteration_instruction)
            messages = messages + [instruction]
            new_messages.append(instruction)

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

            # Return only NEW messages (delta) to avoid add_messages duplication.
            # add_messages reducer will append these to existing state messages.
            state["messages"] = new_messages + [response]

            # Log for debugging
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_names = [tc.get('name', 'unknown') for tc in response.tool_calls]
                logger.info(f"REACT: LLM requested tools: {tool_names}")
            else:
                logger.info("REACT: LLM did not request tools (proceeding to final)")

            # Trace LLM call
            get_trace_logger().log_llm_call(
                agent="react_model",
                input_messages=messages,
                response=response,
                metadata={
                    "iteration": iteration,
                    "has_tool_calls": bool(hasattr(response, 'tool_calls') and response.tool_calls),
                },
            )

        except Exception as e:
            logger.error(f"REACT: LLM invocation failed: {e}")
            # Add error message and continue
            error_msg = AIMessage(content=f"LLM 调用失败: {str(e)}")
            state["messages"] = new_messages + [error_msg]

        return state

    return model_node


def extract_results_node(state: Dict) -> Dict:
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

            elif tool_name == "three_sigma_tool":
                state["three_sigma_result"] = result
                if result.get("success"):
                    anomalies = result.get("anomalies", [])
                    logger.info(f"REACT: 3-sigma found {len(anomalies)} anomalous metrics")
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

    # Return only modified fields, NOT messages.
    # This prevents the add_messages reducer from re-adding existing messages.
    result = {k: v for k, v in state.items() if k != "messages"}
    return result


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
