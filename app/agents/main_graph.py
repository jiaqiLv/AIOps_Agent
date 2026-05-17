"""Main Graph - Single entry point with nested subgraphs

This is the main graph that contains nested subgraphs:
- supervisor_agent: Main orchestrator (expandable in Studio)
- diagnose_agent: ReAct loop root cause analysis subagent (expandable in Studio)

The main graph handles routing between subgraphs based on state.
"""

from typing import Dict, Any, List, Optional, Literal, TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.graph.ui import AnyUIMessage, ui_message_reducer
from app.agents.supervisor_agent import SupervisorAgentState, supervisor_agent as supervisor_subgraph
from app.models.react_agent_state import ReactAgentState
from app.agents.diagnose_agent import diagnose_agent as diagnose_subgraph
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _get_text(content) -> str:
    """Extract plain text from message content, handling Studio's list-of-blocks format."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return " ".join(parts)
    return str(content)


class MainState(TypedDict, total=False):
    """Main graph state with proper add_messages reducer for Studio compatibility."""
    user_input: str
    messages: Annotated[List[BaseMessage], add_messages]
    ui: Annotated[Sequence[AnyUIMessage], ui_message_reducer]
    continue_conversation: bool
    action: str
    csv_file_path: Optional[str]
    inject_time: Optional[float]
    abnormal_kpi: Optional[str]
    gamma: Optional[int]
    alpha: Optional[float]
    dataset_type: Optional[str]
    diagnose_result: Optional[Dict[str, Any]]


def state_to_supervisor(state: MainState, has_diagnose_result: bool = False) -> SupervisorAgentState:
    """Convert main state to supervisor state"""
    return {
        "user_input": state.get("user_input", ""),
        "messages": state.get("messages", []),
        "action": state.get("action", "respond"),
        "csv_file_path": state.get("csv_file_path"),
        "inject_time": state.get("inject_time"),
        "abnormal_kpi": state.get("abnormal_kpi"),
        "gamma": state.get("gamma"),
        "alpha": state.get("alpha"),
        "dataset_type": state.get("dataset_type"),
        "diagnose_result": state.get("diagnose_result") if has_diagnose_result else None,
        "response_message": None,
        "continue_conversation": True
    }


def state_from_supervisor(main_state: MainState, supervisor_state: SupervisorAgentState) -> MainState:
    """Convert supervisor state back to main state"""
    main_state["user_input"] = supervisor_state.get("user_input", "")
    main_state["action"] = supervisor_state.get("action", "respond")
    main_state["csv_file_path"] = supervisor_state.get("csv_file_path")
    main_state["inject_time"] = supervisor_state.get("inject_time")
    main_state["abnormal_kpi"] = supervisor_state.get("abnormal_kpi")
    main_state["gamma"] = supervisor_state.get("gamma")
    main_state["alpha"] = supervisor_state.get("alpha")
    main_state["dataset_type"] = supervisor_state.get("dataset_type")
    main_state["continue_conversation"] = supervisor_state.get("continue_conversation", True)
    # Don't blindly copy messages — the add_messages reducer would duplicate.
    # Only append supervisor's new AIMessages that weren't in the input.
    return main_state


def state_to_diagnose(state: MainState) -> ReactAgentState:
    """Convert main state to diagnose state."""
    task = state.get("user_input", "")
    if not task:
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                task = _get_text(msg.content)
                break

    return {
        "messages": [],
        "task_description": task,
        "csv_file_path": state.get("csv_file_path"),
        "inject_time": state.get("inject_time"),
        "abnormal_kpi": state.get("abnormal_kpi"),
        "gamma": state.get("gamma", 5),
        "alpha": state.get("alpha", 0.05),
        "csv_headers": None,
        "rcd_result": None,
        "pc_result": None,
        "graph_visualization": None,
        "tool_errors": [],
        "integrated_result": None,
    }


def supervisor_node(state: MainState) -> MainState:
    """Node that wraps the supervisor subgraph."""
    logger.info("MAIN: Entering supervisor subgraph")

    # Ensure user_input from messages (for Studio where only messages is set)
    if not state.get("user_input"):
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                state["user_input"] = _get_text(msg.content)
                break

    # Record which messages already exist so we only add new ones
    existing_ids = {getattr(m, "id", None) for m in state.get("messages", [])}

    has_diagnose_result = state.get("diagnose_result") is not None
    supervisor_state = state_to_supervisor(state, has_diagnose_result)
    result = supervisor_subgraph.invoke(supervisor_state)
    state = state_from_supervisor(state, result)

    new_ai = [
        m for m in result.get("messages", [])
        if isinstance(m, AIMessage)
        and m.content
        and getattr(m, "id", None) not in existing_ids
    ]
    if new_ai:
        state["messages"] = state.get("messages", []) + new_ai
        logger.info(f"MAIN: Appended {len(new_ai)} supervisor message(s) to chat")

    logger.info(f"MAIN: Supervisor completed, action={state.get('action')}")
    return state


def diagnose_node(state: MainState) -> MainState:
    """Node that wraps the diagnose subgraph."""
    logger.info("MAIN: Entering diagnose subgraph")

    if not state.get("user_input"):
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                state["user_input"] = _get_text(msg.content)
                break

    logger.info(f"MAIN: Diagnose — user_input='{str(state.get('user_input', ''))[:80]}'")

    diagnose_state = state_to_diagnose(state)

    result = diagnose_subgraph.invoke(diagnose_state)

    # Log the diagnose result for debugging
    logger.info(f"MAIN: Diagnose result keys: {result.keys() if isinstance(result, dict) else 'not a dict'}")
    logger.info(f"MAIN: Diagnose final_response: {result.get('final_response', 'None')[:200] if result.get('final_response') else 'None'}")
    logger.info(f"MAIN: Diagnose messages count: {len(result.get('messages', []))}")

    # Store the full result
    state["diagnose_result"] = result

    # Extract results to main state for supervisor access
    if result.get("csv_file_path"):
        state["csv_file_path"] = result["csv_file_path"]
    if result.get("inject_time"):
        state["inject_time"] = result["inject_time"]
    if result.get("abnormal_kpi"):
        state["abnormal_kpi"] = result["abnormal_kpi"]
    if result.get("gamma"):
        state["gamma"] = result["gamma"]
    if result.get("alpha"):
        state["alpha"] = result["alpha"]

    # Chat 回复由 supervisor 写入 messages（含拓扑图），此处仅保存结果
    state["action"] = "have_diagnose_result"

    logger.info("MAIN: Diagnose completed, will return to supervisor")
    return state


def route_main(state: MainState) -> Literal["supervisor_agent", "diagnose_agent", END]:
    """
    Route function for main graph.
    - "call_diagnose" → diagnose_agent
    - "have_diagnose_result" → supervisor_agent
    - otherwise → END
    """
    action = state.get("action", "")
    if action == "call_diagnose":
        return "diagnose_agent"
    if action == "have_diagnose_result":
        return "supervisor_agent"
    return END


def build_main_graph() -> StateGraph:
    """Build the main graph with nested subgraphs."""
    logger.info("Building main graph with nested subgraphs")

    builder = StateGraph(MainState)

    builder.add_node("supervisor_agent", supervisor_node)
    builder.add_node("diagnose_agent", diagnose_node)

    builder.set_entry_point("supervisor_agent")

    builder.add_conditional_edges(
        "supervisor_agent",
        route_main,
        {
            "diagnose_agent": "diagnose_agent",
            "supervisor_agent": "supervisor_agent",
            END: END
        }
    )

    builder.add_edge("diagnose_agent", "supervisor_agent")

    graph = builder.compile()
    logger.info("Main graph compiled with nested subgraphs")
    return graph


# Create global instance
main_graph = build_main_graph()
