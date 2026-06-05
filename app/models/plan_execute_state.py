"""Plan-Execute State

State schema for the Plan-and-Execute supervisor.
This is a unified state that includes all fields required by sub-agent subgraphs,
enabling compiled subgraphs to be added directly as graph nodes.

Messages uses plain assignment (no add_messages reducer) so the executor can
clear them before each subgraph run, preventing message leakage between steps.
"""

from typing import Dict, Any, List, Optional, TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.ui import AnyUIMessage, ui_message_reducer


class PlanStep(TypedDict, total=False):
    """A single step in the execution plan."""
    step_id: int
    name: str              # 步骤名称（如"异常检测"）
    agent: str             # "detection" | "diagnose" | "report"
    input: Dict[str, Any]  # Planner 为此步定义的输入
    status: str            # "pending" | "running" | "completed" | "failed"
    error: Optional[str]   # 失败时的错误信息


class PlanExecuteState(TypedDict, total=False):
    """Unified state for the Plan-Execute supervisor and its sub-agent subgraphs.

    Contains all fields from DetectionAgentState, ReactAgentState, and
    ReportAgentState so that compiled subgraphs can be added as direct graph
    nodes.  Sub-agent fields are optional (total=False) and only populated
    when the corresponding subgraph runs.

    Messages uses plain List[BaseMessage] (no add_messages reducer) so the
    executor can assign an empty list before each subgraph, giving the
    subgraph a clean message slate.
    """

    # ==================== Core Conversation ====================
    messages: List[BaseMessage]  # Plain assignment — executor clears before each subgraph
    ui: Annotated[Sequence[AnyUIMessage], ui_message_reducer]
    user_input: str
    task_description: str

    # ==================== Plan ====================
    plan: List[PlanStep]
    current_step_index: int     # 0-based, -1 means no plan
    plan_reasoning: str         # LLM's planning reasoning
    plan_reply: str             # LLM's direct reply text (for non-agent queries)

    # ==================== Result Storage ====================
    step_results: Dict[int, Dict[str, Any]]   # step_id → result dict

    # Pending step execution (transient, for executor → subgraph → step_complete flow)
    pending_step_agent: str     # Agent name being executed
    pending_step_id: int        # Step ID being executed

    # ==================== Shared Sub-Agent Input Fields ====================
    # Populated by executor_node via adapter.build_input(), read by subgraphs
    csv_file_path: Optional[str]
    csv_headers: Optional[List[str]]
    inject_time: Optional[float]
    abnormal_kpi: Optional[str]
    max_iterations: int
    iteration_count: int
    tool_errors: List[Dict[str, Any]]
    tool_results: Dict[str, Any]
    gamma: Optional[int]
    alpha: Optional[float]

    # ==================== Detection Output Fields ====================
    three_sigma_result: Optional[Dict[str, Any]]
    bld_metric_result: Optional[Dict[str, Any]]
    anomaly_report: Optional[List[Dict[str, Any]]]
    detection_parameters: Optional[Dict[str, Any]]

    # ==================== Diagnose Output Fields ====================
    rcd_result: Optional[Dict[str, Any]]
    pc_result: Optional[Dict[str, Any]]
    graph_visualizations: Optional[List[Dict[str, Any]]]
    fault_type: Optional[str]
    root_causes: Optional[List[Dict[str, Any]]]
    propagation_path: Optional[List[Dict[str, Any]]]
    integrated_result: Optional[str]

    # ==================== Report-Specific Input Fields ====================
    detection_anomaly_report: Optional[List[Dict[str, Any]]]
    diagnose_fault_type: Optional[str]
    diagnose_root_causes: Optional[List[Dict[str, Any]]]
    diagnose_propagation_path: Optional[List[Dict[str, Any]]]

    # ==================== Interrupt Support ====================
    interrupted: Optional[bool]
    interrupt_data: Optional[Dict[str, Any]]

    # ==================== Multi-turn Context ====================
    # Persists key analysis parameters across turns so the planner can enrich
    # task_description with context (date, metric name, data path).
    session_context: Optional[Dict[str, Any]]

    # ==================== Conversation Log ====================
    # Accumulates messages across subgraph executions.
    # Plain List (no reducer) — subgraphs don't have this field,
    # so it is ignored during subgraph execution and preserved in parent state.
    _conversation_log: Optional[List[BaseMessage]]

    # ==================== Output ====================
    final_response: Optional[str]
    continue_conversation: bool
