"""Prompt-based ReAct Node Implementation

This module implements a ReAct loop where tool descriptions are injected
into the system prompt and the LLM generates text-based tool calls.
This approach doesn't rely on API function calling and is more debuggable.
"""

import json
import re
from typing import Dict, Any, List, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from app.models.react_agent_state import ReactAgentState
from app.utils.logger import get_logger
from app.utils.llm_logger import log_llm_conversation
from app.utils.tool_prompt_generator import generate_diagnose_agent_tools_prompt

logger = get_logger(__name__)


# Tool calling pattern to match in LLM output
TOOL_CALL_PATTERN = r'调用:\s*(\w+)\s*\n参数:\s*\n((?:[ \t]*\w+:\s*[^\n]+\n?)*)'

# Pattern for parsing list parameters (edges, root_causes)
LIST_PATTERN = r'\[\s*[^\]]*\]|"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\''


def parse_tool_call(llm_output: str) -> Optional[Dict[str, Any]]:
    """Parse tool call from LLM text output.

    Args:
        llm_output: Text output from LLM

    Returns:
        Dict with 'tool_name' and 'parameters' if found, None otherwise
    """
    match = re.search(TOOL_CALL_PATTERN, llm_output, re.MULTILINE)
    if match:
        tool_name = match.group(1).strip()
        params_text = match.group(2)

        # Parse parameters with better handling for complex types
        parameters = _parse_parameters(params_text)

        logger.debug(f"Parsed tool call: {tool_name} with {len(parameters)} parameters")
        return {"tool_name": tool_name, "parameters": parameters}

    return None


def _parse_parameters(params_text: str) -> Dict[str, Any]:
    """Parse parameters from tool call text.

    Args:
        params_text: Parameter text block

    Returns:
        Dictionary of parameters
    """
    parameters = {}

    for line in params_text.split('\n'):
        line = line.strip()
        if not line or ':' not in line:
            continue

        # Split on first colon only
        parts = line.split(':', 1)
        if len(parts) < 2:
            continue

        key = parts[0].strip()
        value = parts[1].strip()

        # Handle different value types
        # 1. Lists (edges, root_causes)
        if value.startswith('[') and value.endswith(']'):
            try:
                # Try JSON parse first
                parameters[key] = json.loads(value.replace("'", '"'))
            except json.JSONDecodeError:
                # Manual parsing for simple lists
                inner = value[1:-1].strip()
                if not inner:
                    parameters[key] = []
                else:
                    # Split by comma and clean
                    items = [item.strip().strip('"\'') for item in inner.split(',')]
                    parameters[key] = items

        # 2. Strings (quoted)
        elif (value.startswith('"') and value.endswith('"')) or \
             (value.startswith("'") and value.endswith("'")):
            parameters[key] = value[1:-1]

        # 3. Numbers
        elif value.isdigit():
            parameters[key] = int(value)
        elif value.replace('.', '').isdigit():
            parameters[key] = float(value)

        # 4. Booleans
        elif value.lower() in ('true', 'false'):
            parameters[key] = value.lower() == 'true'

        # 5. Plain strings
        else:
            parameters[key] = value

    return parameters


def execute_tool(tool_name: str, parameters: Dict[str, Any], tools: List[StructuredTool]) -> str:
    """Execute a tool with given parameters.

    Args:
        tool_name: Name of the tool to execute
        parameters: Parameters for the tool
        tools: List of available tools

    Returns:
        JSON string result from tool execution
    """
    tool_map = {tool.name: tool for tool in tools}

    if tool_name not in tool_map:
        return json.dumps({
            "success": False,
            "error": f"Tool '{tool_name}' not found. Available tools: {list(tool_map.keys())}"
        }, ensure_ascii=False)

    tool = tool_map[tool_name]

    try:
        # Invoke the tool
        result = tool.invoke(parameters)

        # If result is not a string, convert to JSON
        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False)

        return result

    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


def create_prompt_based_model_node(llm, tools: List[StructuredTool], system_prompt: str):
    """Create a model node that uses prompt-based tool calling.

    Args:
        llm: LLM instance (without tools bound)
        tools: List of available tools
        system_prompt: Base system prompt

    Returns:
        Node function for StateGraph
    """
    # Inject tools description into system prompt
    tools_prompt = generate_diagnose_agent_tools_prompt()
    enhanced_prompt = f"{system_prompt}\n\n{tools_prompt}"

    logger.info(f"Created prompt-based model node with {len(tools)} tools")

    def model_node(state: ReactAgentState) -> ReactAgentState:
        """Invoke LLM and parse tool calls from text output."""
        iteration = state.get("iteration_count", 0) + 1
        state["iteration_count"] = iteration

        # Prepare messages
        messages = list(state.get("messages", []))

        # If first iteration, create messages from task_description
        if not messages:
            task_description = state.get("task_description", "")
            if task_description:
                messages = [
                    SystemMessage(content=enhanced_prompt),
                    HumanMessage(content=task_description)
                ]
                logger.info(f"REACT: Initial task: {task_description[:100]}...")
            else:
                # No task, just system prompt
                messages = [SystemMessage(content=enhanced_prompt)]
        elif not any(isinstance(m, SystemMessage) for m in messages):
            # Add system prompt if not present
            messages = [SystemMessage(content=enhanced_prompt)] + messages

        logger.info(f"REACT: Iteration {iteration}, invoking LLM")

        try:
            # Invoke LLM
            response = llm.invoke(messages)

            # Handle different response types
            if isinstance(response, str):
                response_text = response
                response_msg = AIMessage(content=response_text)
            elif isinstance(response, AIMessage):
                response_text = response.content
                response_msg = response
            else:
                response_text = str(response)
                response_msg = AIMessage(content=response_text)

            # Check for tool call in response
            tool_call = parse_tool_call(response_text)

            if tool_call:
                # Execute the tool
                tool_name = tool_call["tool_name"]
                parameters = tool_call["parameters"]

                logger.info(f"REACT: Parsed tool call: {tool_name} with params: {parameters}")

                # Execute tool
                tool_result = execute_tool(tool_name, parameters, tools)

                logger.info(f"REACT: Tool result: {tool_result[:200]}...")

                # Create tool call message and result message
                import uuid
                call_id = f"call_{uuid.uuid4().hex[:12]}"

                # Create AIMessage with tool_calls (for compatibility with ToolNode format)
                response_msg.tool_calls = [{
                    "name": tool_name,
                    "args": parameters,
                    "id": call_id,
                    "type": "tool_call",
                }]

                # Add messages to state
                state["messages"] = messages + [response_msg, ToolMessage(
                    content=tool_result,
                    tool_call_id=call_id,
                    name=tool_name
                )]

                # Log
                log_llm_conversation(
                    agent_name="react_model",
                    iteration=iteration,
                    input_messages=messages,
                    response=response_msg,
                    metadata={
                        "has_tool_calls": True,
                        "tool_count": 1,
                        "tool_name": tool_name
                    }
                )

            else:
                # No tool call, this might be the final response
                logger.info("REACT: No tool call detected, treating as potential final response")
                state["messages"] = messages + [response_msg]

                # If this is not the first iteration and we have some results,
                # this might be the final answer
                if iteration > 1:
                    state["final_response"] = response_text
                    state["integrated_result"] = response_text

                log_llm_conversation(
                    agent_name="react_model",
                    iteration=iteration,
                    input_messages=messages,
                    response=response_msg,
                    metadata={
                        "has_tool_calls": False,
                        "tool_count": 0
                    }
                )

        except Exception as e:
            logger.error(f"REACT: LLM invocation failed: {e}")
            error_msg = AIMessage(content=f"系统错误: {str(e)}")
            state["messages"] = messages + [error_msg]

        return state

    return model_node


def create_prompt_based_route_function():
    """Create routing function for prompt-based ReAct loop."""
    def route_after_model(state: Dict) -> str:
        """Route after model invocation."""
        messages = state.get("messages", [])
        if not messages:
            return "final"

        last_message = messages[-1]

        # Check if last message has tool_calls
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            # Check iteration limit
            iteration = state.get("iteration_count", 0)
            max_iterations = state.get("max_iterations", 10)
            if iteration >= max_iterations:
                return "final"
            return "tools"

        # Check if we have a final response
        if state.get("final_response"):
            return "final"

        # If first iteration with no tool calls, try again
        if state.get("iteration_count", 0) == 1:
            return "model"  # Try again

        return "final"

    return route_after_model
