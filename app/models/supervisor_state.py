"""Supervisor Agent State Schema"""

from typing import Any, Dict, List, Optional, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class ToolCallRequest(Dict[str, Any]):
    """Tool call request structure"""
    tool_name: str
    arguments: Dict[str, Any]
    missing_fields: List[str]


class SupervisorState(Dict[str, Any]):
    """
    State for Supervisor Agent.

    This state is used across the supervisor agent's workflow,
    tracking user input, tool calls, and conversation state.
    """
    # Core conversation state
    messages: Annotated[List[BaseMessage], add_messages]

    # User input
    user_input: str

    # Tool selection
    selected_tool: Optional[str]
    tool_call_request: Optional[ToolCallRequest]

    # Extracted parameters for diagnosis
    extracted_params: Dict[str, Any]
    missing_fields: List[str]

    # Human-in-the-loop
    need_human_input: bool
    human_question: Optional[str]
    human_feedback: Optional[str]

    # Diagnosis result from subagent
    diagnose_result: Optional[Dict[str, Any]]

    # Final response
    final_response: Optional[str]
    error: Optional[str]

    # Iteration control
    iteration_count: int
    max_iterations: int

    # Continue conversation flag
    continue_conversation: bool

    # Action for routing
    action: str  # "call_diagnose", "respond", "ask_path", "have_diagnose_result"

    # Diagnostic parameters (passed to diagnose agent)
    csv_file_path: Optional[str]
    inject_time: Optional[float]
    fault_injection_time: Optional[str]
    abnormal_kpi: Optional[str]
    gamma: Optional[int]
    alpha: Optional[float]
    dataset_type: Optional[str]

    # Path components
    benchmark: Optional[str]
    instance: Optional[str]
    case: Optional[str]
