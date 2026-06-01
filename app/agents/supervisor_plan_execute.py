"""Supervisor Agent - Plan-and-Execute Architecture

Replaces the old ReAct-based supervisor. The new supervisor:
1. Planner: Analyzes user request, generates an execution plan (list of steps)
2. Executor: Iterates through plan steps, invoking subgraphs via adapters
3. Finalize: Extracts report result, generates HTML, attaches topology
4. Direct Reply: For simple conversational responses (no sub-agent needed)

Graph structure:
START → planner(LLM生成计划)
            │
            ├── empty plan / direct_reply → direct_reply → END
            │
            └── has steps → executor(调度执行) ──→ executor(循环) ──→ finalize → END
"""

import json
import re
from typing import Dict, Any, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END

from app.models.plan_execute_state import PlanExecuteState, PlanStep
from app.config.model_config import get_deepseek_llm
from app.agents.subgraph_registry import get_adapter, get_subgraph
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
                user_input = str(msg.content) if isinstance(msg.content, str) else str(msg.content)
                break

    # Load planner prompt
    system_prompt = load_prompt("app/prompts/supervisor_planner.md")

    # Build messages for LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    try:
        llm = get_deepseek_llm(temperature=0)
        response = llm.invoke(messages)

        # Trace LLM call
        get_trace_logger().log_llm_call(
            agent="supervisor_planner",
            input_messages=messages,
            response=response,
            metadata={"user_input": user_input[:200] if user_input else ""},
        )

        # Extract text content
        content = response if isinstance(response, str) else str(response)
        # Handle AIMessage objects
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, list):
                # Studio format: list of blocks
                content = " ".join(
                    block.get("text", "") for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )

        # Parse JSON from response (may be wrapped in markdown code block)
        plan_data = _parse_plan_json(content)

        if plan_data is None:
            logger.warning("PLANNER: Failed to parse plan JSON, falling back to direct_reply")
            state["plan"] = []
            state["plan_reasoning"] = content
            state["plan_reply"] = ""
            state["current_step_index"] = -1
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

        logger.info(f"PLANNER: Generated plan with {len(steps)} steps")
        for step in steps:
            logger.info(f"  Step {step['step_id']}: {step['name']} (agent={step['agent']})")

    except Exception as e:
        logger.error(f"PLANNER: Failed to generate plan: {e}")
        state["plan"] = []
        state["plan_reasoning"] = f"Planning failed: {e}"
        state["plan_reply"] = ""
        state["current_step_index"] = -1

    return state


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


# ==================== Executor Node ====================

def executor_node(state: PlanExecuteState) -> PlanExecuteState:
    """Execute the current plan step by invoking the appropriate subgraph.

    Uses the SubAgentAdapter interface for generic dispatch:
      adapter.build_input(step.input, step_results) → subgraph.invoke() → adapter.extract_result()
    """
    plan = state.get("plan", [])
    idx = state.get("current_step_index", -1)

    if idx < 0 or idx >= len(plan):
        logger.warning(f"EXECUTOR: Invalid step index {idx}")
        return state

    step = plan[idx]
    step["status"] = "running"

    logger.info(f"EXECUTOR: Running step {step['step_id']}: {step['name']} (agent={step['agent']})")

    try:
        adapter = get_adapter(step["agent"])
        subgraph = get_subgraph(step["agent"])

        # Set agent name for tool tracing
        from app.tools.langchain_tool_adapters import set_agent_name
        set_agent_name(step["agent"])

        # Build input using adapter
        step_results = state.get("step_results", {})
        input_state = adapter.build_input(step.get("input", {}), step_results)

        # Invoke subgraph
        output = subgraph.invoke(input_state)

        # Extract result using adapter
        result = adapter.extract_result(output)

        # Store in step_results
        if "step_results" not in state:
            state["step_results"] = {}
        state["step_results"][step["step_id"]] = result

        step["status"] = "completed" if result.get("success") else "failed"
        if not result.get("success"):
            step["error"] = result.get("summary", "未知错误")

        logger.info(f"EXECUTOR: Step {step['step_id']} completed (success={result.get('success')})")

    except Exception as e:
        logger.error(f"EXECUTOR: Step {step['step_id']} failed: {e}")
        step["status"] = "failed"
        step["error"] = str(e)

        if "step_results" not in state:
            state["step_results"] = {}
        state["step_results"][step["step_id"]] = {
            "success": False,
            "summary": f"步骤执行失败: {e}",
        }

    state["current_step_index"] = idx + 1
    return state


def route_after_executor(state: PlanExecuteState) -> str:
    """Route after executor: loop if more steps, finalize if done."""
    plan = state.get("plan", [])
    idx = state.get("current_step_index", -1)

    if idx < len(plan):
        return "executor"
    return "finalize"


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

    state["messages"] = [AIMessage(content=state["final_response"])]
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

    state["messages"] = [final_msg]
    state["continue_conversation"] = True
    logger.info("FINALIZE: Final response built")
    return state


# ==================== Graph Builder ====================

def build_plan_execute_agent():
    """Build the Plan-Execute supervisor agent.

    Graph: START → planner → [direct_reply | executor → ... → finalize] → END

    Returns:
        Compiled StateGraph
    """
    logger.info("Building Plan-Execute supervisor agent")

    builder = StateGraph(PlanExecuteState)

    # Add nodes
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("direct_reply", direct_reply_node)
    builder.add_node("finalize", finalize_node)

    # Set entry point
    builder.set_entry_point("planner")

    # Add conditional edges
    builder.add_conditional_edges("planner", route_after_planner, {
        "executor": "executor",
        "direct_reply": "direct_reply",
    })
    builder.add_conditional_edges("executor", route_after_executor, {
        "executor": "executor",   # loop
        "finalize": "finalize",   # plan complete
    })

    # Terminal edges
    builder.add_edge("direct_reply", END)
    builder.add_edge("finalize", END)

    graph = builder.compile()
    logger.info("Plan-Execute supervisor agent compiled")
    return graph


# Lazy-loaded graphs (built on first access, not at import time)
# - graph: for langgraph dev (module-level variable expected by CLI)
# - plan_execute_agent: for programmatic use
graph = LazyGraph(build_plan_execute_agent)
plan_execute_agent = graph
