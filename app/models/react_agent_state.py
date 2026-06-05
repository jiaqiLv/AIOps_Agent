"""Generic ReAct Loop Agent State

This state schema is designed for configuration-driven ReAct loops where:
- The LLM autonomously decides which tools to call
- Tool results are stored dynamically
- Human-in-the-loop is supported via interrupts
- Errors are tracked but don't stop execution
"""

from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langchain_core.messages import BaseMessage


def add_messages(left: List[BaseMessage], right: List[BaseMessage]) -> List[BaseMessage]:
    """Add messages reducer for langgraph state"""
    if not right:
        return left
    return left + right


class ReactAgentState(TypedDict, total=False):
    """Generic ReAct Loop State for configuration-driven agents

    This state schema supports:
    - Dynamic tool invocation (LLM decides which tools to call)
    - Flexible result storage (key-value pairs for any tool output)
    - Iteration control with max iterations
    - Error tracking without stopping execution
    - Human-in-the-loop interrupts
    """

    # ==================== Core Conversation ====================
    messages: Annotated[List[BaseMessage], add_messages]
    """Message history including HumanMessage, AIMessage, ToolMessage"""

    task_description: str
    """Original task description from user"""

    # ==================== Tool Results Storage ====================
    tool_results: Dict[str, Any]
    """Generic storage for tool outputs.
    Keys are tool names, values are parsed results.
    Examples:
    {
        "csv_reader_tool": {"shape": [1000, 50], "columns": [...]},
        "rcd_tool": {"root_causes": [...], "success": True},
        "pc_tool": {"edges": [...], "root_causes": [...]}
    }
    """

    # ==================== Execution Control ====================
    iteration_count: int
    """Current iteration number (starts at 0, increments each model call)"""

    max_iterations: int
    """Maximum iterations before forcing completion (default: 10)"""

    tool_errors: List[Dict[str, Any]]
    """List of tool execution errors.
    Each entry: {"tool": str, "error": str, "iteration": int}
    Errors don't stop execution; LLM can retry, skip, or ask user.
    """

    # ==================== Interrupt Handling ====================
    interrupted: bool
    """Whether an interrupt was triggered (e.g., ask_user called)"""

    interrupt_data: Optional[Dict[str, Any]]
    """Data associated with the interrupt.
    Example: {"question": "What is the inject_time?", "tool": "rcd_tool"}
    """

    # ==================== Final Output ====================
    final_response: Optional[str]
    """Final synthesized response from the agent"""

    # ==================== Legacy Compatibility ====================
    # These fields are kept for compatibility with existing code
    csv_file_path: Optional[str]
    """Resolved CSV file path (if CSV was loaded)"""

    csv_headers: Optional[List[str]]
    """CSV column headers (if CSV was loaded)"""

    inject_time: Optional[float]
    """Parsed inject_time as Unix timestamp (if provided)"""

    abnormal_kpi: Optional[str]
    """Abnormal KPI name (if provided)"""

    gamma: int
    """IAF-RCL algorithm gamma parameter (default: 5)"""

    alpha: float
    """KE-FPC algorithm alpha parameter (default: 0.05)"""

    three_sigma_result: Optional[Dict[str, Any]]
    """3-sigma anomaly detection result (legacy, also in tool_results["three_sigma_tool"])"""

    bld_metric_result: Optional[Dict[str, Any]]
    """BLD Metric (ECOD) anomaly detection result"""

    rcd_result: Optional[Dict[str, Any]]
    """IAF-RCL algorithm result (legacy, also in tool_results["rcd_tool"])"""

    pc_result: Optional[Dict[str, Any]]
    """KE-FPC algorithm result (legacy, also in tool_results["pc_tool"])"""

    integrated_result: Optional[str]
    """Synonym for final_response (legacy compatibility)"""

    graph_visualizations: Optional[List[Dict[str, Any]]]
    """List of graph visualization results from graph_visualization_tool"""

    # ==================== Structured Diagnose Output ====================
    fault_type: Optional[str]
    """Inferred fault type (e.g. CPU_RESOURCE_SATURATION, MEMORY_LEAK)"""

    root_causes: Optional[List[Dict[str, Any]]]
    """Structured root cause records with metric, confidence, algorithms, reason"""

    propagation_path: Optional[List[Dict[str, Any]]]
    """Causal propagation edges from pc_tool"""
