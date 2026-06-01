"""Plan-Execute State

State schema for the Plan-and-Execute supervisor.
Replaces the old SupervisorReActState with a plan-driven workflow:
  planner → executor → reporter
"""

from typing import Dict, Any, List, Optional, TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
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
    """State for the Plan-Execute supervisor.

    Replaces the ReAct-based SupervisorReActState. Key differences:
    - Uses `plan: List[PlanStep]` instead of LLM-driven tool calls
    - Uses `step_results: Dict[int, Dict]` as generic memory instead of
      `detection_result` / `diagnose_result` separate fields
    - Executor iterates through plan steps, invoking subgraphs via adapters
    """
    # Core conversation
    messages: Annotated[List[BaseMessage], add_messages]
    ui: Annotated[Sequence[AnyUIMessage], ui_message_reducer]
    user_input: str
    task_description: str

    # Plan
    plan: List[PlanStep]
    current_step_index: int     # 0-based, -1 means no plan
    plan_reasoning: str         # LLM's planning reasoning
    plan_reply: str             # LLM's direct reply text (for non-agent queries)

    # Generic result storage (corresponds to case.md's memory dict)
    step_results: Dict[int, Dict[str, Any]]   # step_id → result dict

    # Output
    final_response: Optional[str]
    continue_conversation: bool
