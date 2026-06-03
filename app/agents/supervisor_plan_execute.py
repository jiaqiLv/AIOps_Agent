"""Supervisor Agent - Plan-and-Execute Architecture

Uses compiled subgraphs as direct graph nodes, making each sub-agent
clickable in LangGraph Studio to reveal its internal structure.

Graph structure:
START → planner(LLM生成计划)
            │
            ├── empty plan / direct_reply → direct_reply → END
            │
            └── has steps → executor(准备输入) → step_router:
                               ├── detection(子图) → step_complete → executor → ...
                               ├── diagnose(子图) → step_complete → executor → ...
                               └── report  (子图) → step_complete → finalize → END

To add a new sub-agent:
  1. Add adapter in subgraph_registry.py
  2. Add build_xxx_agent() in the agent module
  3. Add entry in _build_subgraphs() below
  4. Add state fields to PlanExecuteState if needed
"""

import json
import re
from typing import Dict, Any, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END

from app.models.plan_execute_state import PlanExecuteState, PlanStep
from app.config.model_config import get_deepseek_llm
from app.agents.subgraph_registry import get_adapter
from app.utils.logger import get_logger
from app.utils.prompt_loader import load_prompt
from app.utils.llm_logger import get_trace_logger
from app.utils.lazy_graph import LazyGraph

logger = get_logger(__name__)


# ==================== Planner Node ====================

def planner_node(state: PlanExecuteState) -> PlanExecuteState:
    """Analyze user request and generate an execution plan.

    Calls LLM (temperature=0, no bind_tools) with the planner prompt,
    parses the JSON response into a List[PlanStep].
    Falls back to direct_reply on parse failure.
    """
    logger.info("PLANNER: Generating execution plan")

    user_input = state.get("user_input", "")
    if not user_input:
        # Try to extract from messages
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                user_input = str(msg.content)
                break

    # Load planner prompt
    system_prompt = load_prompt("app/prompts/supervisor_planner.md")

    # Build messages for LLM
    messages = [
        {"role": "system", "content": system_prompt},
    ]

    # Inject session context as synthetic conversation history
    session_context = state.get("session_context")
    if session_context and session_context.get("csv_file_path"):
        context_summary = _format_session_context(session_context)
        messages.append({"role": "user", "content": context_summary})
        messages.append({"role": "assistant", "content": "收到，已记录分析上下文。请问您还需要什么帮助？"})

    messages.append({"role": "user", "content": user_input})

    try:
        llm = get_deepseek_llm(temperature=0)

        # M1: LLM retry + circuit breaker
        from app.middleware.llm_error_handling import (
            get_llm_retry_handler, LLMCircuitBreakerError, LLMMaxRetriesError,
        )
        response = get_llm_retry_handler().invoke(llm, messages)

        # M6: Token usage tracking
        from app.middleware.token_usage import get_token_tracker
        get_token_tracker().track("supervisor_planner", response)

        # Trace LLM call
        get_trace_logger().log_llm_call(
            agent="supervisor_planner",
            input_messages=messages,
            response=response,
            metadata={"user_input": user_input[:200] if user_input else ""},
        )

        # Extract text content
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, list):
                # Studio format: list of blocks
                content = " ".join(
                    block.get("text", "") for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
        else:
            content = str(response)

        # Parse JSON from response (may be wrapped in markdown code block)
        plan_data = _parse_plan_json(content)

        if plan_data is None:
            logger.warning("PLANNER: Failed to parse plan JSON, falling back to direct_reply")
            state["plan"] = []
            state["plan_reasoning"] = content
            state["plan_reply"] = ""
            state["current_step_index"] = -1
            state["_conversation_log"] = list(state.get("messages", []))
            return state

        steps = []
        for raw_step in plan_data.get("steps", []):
            step = PlanStep(
                step_id=raw_step["id"],
                name=raw_step.get("name", ""),
                agent=raw_step.get("agent", "direct_reply"),
                input=raw_step.get("input", {}),
                status="pending",
                error=None,
            )
            steps.append(step)

        state["plan"] = steps
        state["plan_reasoning"] = plan_data.get("reasoning", "")
        state["plan_reply"] = plan_data.get("reply", "")
        state["current_step_index"] = 0 if steps else -1

        # Initialize conversation log with initial messages
        state["_conversation_log"] = list(state.get("messages", []))

        logger.info(f"PLANNER: Generated plan with {len(steps)} steps")
        for step in steps:
            logger.info(f"  Step {step['step_id']}: {step['name']} (agent={step['agent']})")

    except Exception as e:
        logger.error(f"PLANNER: Failed to generate plan: {e}")
        state["plan"] = []
        state["plan_reasoning"] = f"Planning failed: {e}"
        state["plan_reply"] = ""
        state["current_step_index"] = -1
        state["_conversation_log"] = list(state.get("messages", []))

    return state


def _format_session_context(ctx: dict) -> str:
    """Format session context as structured summary for the planner LLM."""
    parts = ["## 历史分析上下文\n以下是前一轮分析的关键参数："]
    if ctx.get("csv_file_path"):
        parts.append(f"- 数据文件: {ctx['csv_file_path']}")
    if ctx.get("inject_time"):
        ts = ctx["inject_time"]
        if isinstance(ts, (int, float)):
            from app.utils.time_utils import format_unix_ts
            ts = format_unix_ts(ts)
        parts.append(f"- 故障注入时间: {ts}")
    if ctx.get("abnormal_kpi"):
        parts.append(f"- 异常 KPI: {ctx['abnormal_kpi']}")
    if ctx.get("detection_summary"):
        parts.append(f"- 检测结果: {ctx['detection_summary']}")
    if ctx.get("completed_agents"):
        names = {"detection": "异常检测", "diagnose": "根因推理", "report": "报告生成"}
        completed = ", ".join(names.get(a, a) for a in ctx["completed_agents"])
        parts.append(f"- 已完成步骤: {completed}")
    return "\n".join(parts)


def _parse_plan_json(content: str) -> Optional[dict]:
    """Parse JSON from planner LLM response.

    Handles cases where JSON is wrapped in markdown code blocks.
    """
    # Try to extract JSON from markdown code block
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)```", content, re.DOTALL)
    if json_match:
        content = json_match.group(1).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the content
    brace_start = content.find("{")
    brace_end = content.rfind("}")
    if brace_start != -1 and brace_end != -1:
        try:
            return json.loads(content[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass

    return None


# ==================== Routing ====================

def route_after_planner(state: PlanExecuteState) -> str:
    """Route after planner: direct_reply if no steps, executor otherwise."""
    plan = state.get("plan", [])
    if not plan or state.get("current_step_index", -1) < 0:
        return "direct_reply"
    return "executor"


def step_router(state: PlanExecuteState) -> str:
    """Route from executor to the appropriate compiled subgraph node."""
    return state.get("pending_step_agent", "")


def route_after_step(state: PlanExecuteState) -> str:
    """Route after step_complete: loop to executor if more steps, else finalize."""
    plan = state.get("plan", [])
    idx = state.get("current_step_index", -1)

    if idx < len(plan):
        return "executor"
    return "finalize"


# ==================== Executor Node ====================

def executor_node(state: PlanExecuteState) -> PlanExecuteState:
    """Prepare input for the current plan step by writing subgraph fields
    directly into the unified state.  Clears messages so the subgraph
    starts with a clean slate.
    """
    plan = state.get("plan", [])
    idx = state.get("current_step_index", -1)

    if idx < 0 or idx >= len(plan):
        logger.warning(f"EXECUTOR: Invalid step index {idx}")
        return state

    step = plan[idx]
    step["status"] = "running"

    logger.info(f"EXECUTOR: Preparing step {step['step_id']}: {step['name']} (agent={step['agent']})")

    adapter = get_adapter(step["agent"])
    step_results = state.get("step_results", {})
    input_state = adapter.build_input(step.get("input", {}), step_results)

    # Write subgraph input fields directly into state (including messages=[])
    state.update(input_state)
    state["pending_step_agent"] = step["agent"]
    state["pending_step_id"] = step["step_id"]

    # Set agent name for tool tracing
    try:
        from app.tools.langchain_tool_adapters import set_agent_name
        set_agent_name(step["agent"])
    except ImportError:
        pass

    return state


# ==================== Step Complete Node ====================

def step_complete_node(state: PlanExecuteState) -> PlanExecuteState:
    """Extract results from the subgraph output (now in the unified state)
    and store them in step_results.
    """
    idx = state.get("current_step_index", -1)
    step_id = state.get("pending_step_id", -1)
    agent = state.get("pending_step_agent", "")

    adapter = get_adapter(agent)
    result = adapter.extract_result(state)

    step_results = dict(state.get("step_results", {}))
    step_results[step_id] = result

    plan = state.get("plan", [])
    if 0 <= idx < len(plan):
        plan[idx]["status"] = "completed" if result.get("success") else "failed"
        if not result.get("success"):
            plan[idx]["error"] = result.get("summary", "未知错误")

    logger.info(f"STEP_COMPLETE: {agent} step {step_id} done (success={result.get('success')})")

    # Accumulate subgraph messages into conversation log
    log = list(state.get("_conversation_log") or [])
    current_msgs = state.get("messages", [])
    if current_msgs:
        log.extend(current_msgs)

    # Add step summary
    log.append(AIMessage(content=f"--- {agent} step {step_id} completed ---"))

    # Return minimal update — preserve plan/step_results, clear messages for next step
    return {
        "step_results": step_results,
        "current_step_index": idx + 1,
        "messages": [],
        "_conversation_log": log,
    }


# ==================== Direct Reply Node ====================

def direct_reply_node(state: PlanExecuteState) -> PlanExecuteState:
    """Handle simple conversational responses (no sub-agent needed).

    Uses the planner's `reply` field from the parsed JSON as the response.
    Falls back to a generic greeting if no reply was provided.
    """
    logger.info("DIRECT_REPLY: Generating direct response")

    reply = state.get("plan_reply", "")

    if reply:
        state["final_response"] = reply
    else:
        state["final_response"] = "您好！我是 AIOps 根因分析助手。请描述您的异常事件或提供数据文件，我将为您进行根因分析。"

    log = list(state.get("_conversation_log") or [])
    state["messages"] = log + [AIMessage(content=state["final_response"])]
    state["continue_conversation"] = True
    logger.info(f"DIRECT_REPLY: Response generated")
    return state


# ==================== Finalize Node ====================

def finalize_node(state: PlanExecuteState) -> PlanExecuteState:
    """Finalize execution: extract report result, generate HTML, attach topology.

    Runs after all plan steps (including report step) are complete.
    Collects results from step_results and builds the final response message.
    """
    logger.info("FINALIZE: Building final response from step results")

    step_results = state.get("step_results", {})
    plan = state.get("plan", [])

    # Collect data from step_results by agent type
    report_result = None
    detection_data = {}
    diagnose_data = {}
    graph_visualizations = []

    for step in plan:
        result = step_results.get(step["step_id"], {})
        if step["agent"] == "report":
            report_result = result
        elif step["agent"] == "detection":
            detection_data = result
        elif step["agent"] == "diagnose":
            diagnose_data = result
            graph_visualizations = result.get("graph_visualizations", [])

    # Set final_response from report step result
    if report_result and report_result.get("success") and report_result.get("summary"):
        state["final_response"] = report_result["summary"]
    else:
        # Fallback: concatenate structured data directly
        parts = []
        if detection_data.get("anomaly_report"):
            from app.utils.prompt_template import format_detection_structured
            parts.append(format_detection_structured(
                detection_data["anomaly_report"],
                detection_data.get("detection_parameters"),
            ))
        if diagnose_data.get("root_causes"):
            from app.utils.prompt_template import format_diagnose_structured
            parts.append(format_diagnose_structured(
                diagnose_data.get("fault_type"),
                diagnose_data.get("root_causes"),
                diagnose_data.get("propagation_path"),
            ))
        if diagnose_data.get("summary"):
            parts.append(diagnose_data["summary"])
        if report_result and report_result.get("summary"):
            parts.append(report_result["summary"])
        state["final_response"] = "\n\n---\n\n".join(parts) if parts else "分析完成，但未生成报告。"

    # Attach topology visualization message if available
    graph_viz = None
    if graph_visualizations:
        graph_viz = graph_visualizations[-1]

    # Generate self-contained HTML report
    report_url = ""
    task_description = state.get("task_description", "") or state.get("user_input", "")
    try:
        from app.utils.topology_chat import generate_html_report
        report_path = generate_html_report(
            report_text=state["final_response"],
            graph_viz=graph_viz,
            detection_result=detection_data if detection_data else None,
            task_description=task_description,
        )
        from app.http_app import get_report_url
        report_url = get_report_url()
        logger.info(f"FINALIZE: HTML report generated at {report_path} ({report_url})")
    except Exception as e:
        logger.warning(f"FINALIZE: HTML report generation failed: {e}")

    from app.utils.topology_chat import build_final_report_message
    final_msg = build_final_report_message(state["final_response"], graph_viz)

    # Append report link to message content
    if report_url:
        link_suffix = f"\n\n---\n\n**完整报告（含图片）**: [{report_url}]({report_url})"
        existing = final_msg.content if isinstance(final_msg.content, str) else str(final_msg.content)
        final_msg = AIMessage(content=existing + link_suffix)

    # Build final messages from conversation log
    log = list(state.get("_conversation_log") or [])
    log.append(final_msg)
    state["messages"] = log
    state["continue_conversation"] = True
    logger.info("FINALIZE: Final response built")
    return state


# ==================== Subgraph Builder ====================

def _build_subgraphs() -> Dict[str, object]:
    """Build all compiled subgraphs and return {name: compiled_graph}.

    To add a new sub-agent, add an entry here and ensure:
      - Adapter exists in subgraph_registry.py
      - State fields added to PlanExecuteState
    """
    from app.agents.detection_agent import build_detection_agent
    from app.agents.diagnose_agent import build_diagnose_agent
    from app.agents.report_agent import build_report_agent

    return {
        "detection": build_detection_agent(),
        "diagnose": build_diagnose_agent(),
        "report": build_report_agent(),
    }


# ==================== Graph Builder ====================

def build_plan_execute_agent():
    """Build the Plan-Execute supervisor agent.

    Sub-agents are registered as compiled subgraph nodes, making them
    expandable in LangGraph Studio (click to see internal ReAct loop).

    Returns:
        Compiled StateGraph
    """
    logger.info("Building Plan-Execute supervisor agent")

    builder = StateGraph(PlanExecuteState)

    # Core nodes
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("step_complete", step_complete_node)
    builder.add_node("direct_reply", direct_reply_node)
    builder.add_node("finalize", finalize_node)

    # Register compiled subgraphs as nodes
    subgraphs = _build_subgraphs()
    step_router_map = {}
    for agent_name, subgraph in subgraphs.items():
        builder.add_node(agent_name, subgraph)
        builder.add_edge(agent_name, "step_complete")
        step_router_map[agent_name] = agent_name

    # Set entry point
    builder.set_entry_point("planner")

    # Planner routing
    builder.add_conditional_edges("planner", route_after_planner, {
        "executor": "executor",
        "direct_reply": "direct_reply",
    })

    # Executor → subgraph routing
    builder.add_conditional_edges("executor", step_router, step_router_map)

    # Step complete → loop or finalize
    builder.add_conditional_edges("step_complete", route_after_step, {
        "executor": "executor",
        "finalize": "finalize",
    })

    # Terminal edges
    builder.add_edge("direct_reply", END)
    builder.add_edge("finalize", END)

    graph = builder.compile()
    logger.info(f"Plan-Execute supervisor agent compiled with subgraphs: {list(subgraphs.keys())}")
    return graph


# Lazy-loaded graphs (built on first access, not at import time)
# - graph: for langgraph dev (module-level variable expected by CLI)
# - plan_execute_agent: for programmatic use
graph = LazyGraph(build_plan_execute_agent)
plan_execute_agent = graph
