"""Main Graph - Single supervisor node wrapper

The main graph wraps the Plan-Execute supervisor. The supervisor
generates an execution plan and dispatches to sub-agents (detection,
diagnose) via a plan-execute loop.

Graph structure:
START → supervisor → END
"""

from typing import Dict, Any, List, Optional, TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.graph.ui import AnyUIMessage, ui_message_reducer

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
    # These fields are kept for backward compatibility with LangGraph Studio
    action: str
    csv_file_path: Optional[str]
    inject_time: Optional[float]
    abnormal_kpi: Optional[str]
    diagnose_result: Optional[Dict[str, Any]]
    # New structured fields from sub-agents
    anomaly_report: Optional[List[Dict[str, Any]]]
    fault_type: Optional[str]
    root_causes: Optional[List[Dict[str, Any]]]
    propagation_path: Optional[List[Dict[str, Any]]]
    graph_visualizations: Optional[List[Dict[str, Any]]]
    # Multi-turn context memory
    session_context: Optional[Dict[str, Any]]


def supervisor_node(state: MainState) -> MainState:
    """Node that wraps the Plan-Execute supervisor subgraph.

    Converts MainState to PlanExecuteState, invokes the supervisor,
    and maps results back to MainState.
    """
    from app.agents.supervisor_plan_execute import plan_execute_agent
    from app.models.plan_execute_state import PlanExecuteState

    logger.info("MAIN: Entering Plan-Execute supervisor")

    # Per-turn cleanup for Studio compatibility (CLI does this outside the graph)
    from app.tools.langchain_tool_adapters import clear_csv_cache
    clear_csv_cache()

    # Always extract user_input from the latest HumanMessage.
    # In Studio, state persists between turns but user_input is a plain
    # field (no reducer), so it stays stale from the previous turn.
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            state["user_input"] = _get_text(msg.content)
            break

    user_input = state.get("user_input", "")

    # Record which messages already exist so we only add new ones
    existing_ids = {getattr(m, "id", None) for m in state.get("messages", [])}

    # Convert to PlanExecuteState
    supervisor_state: PlanExecuteState = {
        "messages": state.get("messages", []),
        "ui": state.get("ui", []),
        "user_input": user_input,
        "task_description": user_input,
        "plan": [],
        "current_step_index": -1,
        "plan_reasoning": "",
        "step_results": {},
        "final_response": None,
        "continue_conversation": True,
        "session_context": state.get("session_context"),
    }

    # Invoke supervisor
    result = plan_execute_agent.invoke(supervisor_state)

    # Map results back to main state
    state["continue_conversation"] = result.get("continue_conversation", True)
    state["final_response"] = result.get("final_response")
    state["action"] = "respond"

    # Propagate sub-agent results and update session context in one pass
    step_results = result.get("step_results", {})
    session_context = dict(state.get("session_context") or {})
    completed_agents = session_context.get("completed_agents", [])

    for step in result.get("plan", []):
        sr = step_results.get(step["step_id"], {})
        if step["agent"] == "detection":
            for key in ("csv_file_path", "inject_time", "abnormal_kpi"):
                if sr.get(key):
                    state[key] = sr[key]
            if sr.get("anomaly_report"):
                state["anomaly_report"] = sr["anomaly_report"]
            # Session context
            if sr.get("success"):
                session_context.update({
                    "csv_file_path": sr.get("csv_file_path"),
                    "inject_time": sr.get("inject_time"),
                    "abnormal_kpi": sr.get("abnormal_kpi"),
                    "detection_summary": (sr.get("summary") or "")[:200],
                })
                if "detection" not in completed_agents:
                    completed_agents.append("detection")
        elif step["agent"] == "diagnose":
            state["diagnose_result"] = sr
            for key in ("csv_file_path", "inject_time", "abnormal_kpi"):
                if sr.get(key):
                    state[key] = sr[key]
            for key in ("fault_type", "root_causes", "propagation_path", "graph_visualizations"):
                if sr.get(key):
                    state[key] = sr[key]
            # Session context
            if sr.get("success"):
                session_context.update({
                    "diagnose_summary": (sr.get("summary") or "")[:200],
                })
                if "diagnose" not in completed_agents:
                    completed_agents.append("diagnose")

    session_context["completed_agents"] = completed_agents
    state["session_context"] = session_context

    # Propagate ALL new messages (AIMessage, ToolMessage, etc.) so Studio shows
    # the full intermediate process — tool calls, tool results, and AI reasoning.
    new_msgs = [
        m for m in result.get("messages", [])
        if getattr(m, "id", None) not in existing_ids
        and not isinstance(m, HumanMessage)  # skip HumanMessage (already in state)
    ]
    if new_msgs:
        state["messages"] = new_msgs
        logger.info(f"MAIN: Propagated {len(new_msgs)} messages to chat "
                     f"(tools={sum(1 for m in new_msgs if isinstance(m, ToolMessage))}, "
                     f"ai={sum(1 for m in new_msgs if isinstance(m, AIMessage))})")

    # If no messages were propagated but we have a final_response, add it
    if not new_msgs and result.get("final_response"):
        state["messages"] = [AIMessage(content=result["final_response"])]

    logger.info(f"MAIN: Supervisor completed, action={state.get('action')}")
    return state


def build_main_graph() -> StateGraph:
    """Build the main graph with a single supervisor node."""
    logger.info("Building main graph (single supervisor node)")

    builder = StateGraph(MainState)

    builder.add_node("supervisor", supervisor_node)

    builder.set_entry_point("supervisor")
    builder.add_edge("supervisor", END)

    graph = builder.compile()
    logger.info("Main graph compiled")
    return graph


# Create global instance
main_graph = build_main_graph()
