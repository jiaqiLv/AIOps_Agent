"""Agent Registry — config-driven builder for ReAct agents.

Reads ``app/config/agents.yaml`` and constructs a compiled LangGraph
StateGraph using the generic factories from ``react_nodes.py``.

Public API:
    load_agent_config(name) -> dict   # raw config from YAML
    build_react_agent(config) -> CompiledStateGraph
"""

import importlib
from typing import Dict, Any, Optional

import yaml
from langgraph.graph import StateGraph, END

from app.agents.nodes.react_nodes import (
    create_model_node,
    extract_results_node,
    route_after_model,
)
from app.utils.logger import get_logger
from app.utils.prompt_loader import load_prompt

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _resolve_state_schema(dotted_path: str):
    """Resolve ``'pkg.module.ClassName'`` to the actual class."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def load_agent_config(agent_name: str) -> Dict[str, Any]:
    """Load and return the config dict for *agent_name* from agents.yaml.

    The ``state_schema`` string is resolved to the real class and stored
    under ``state_schema`` in the returned dict.
    """
    import os
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config", "agents.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    agents = cfg.get("agents", {})
    if agent_name not in agents:
        raise ValueError(f"Agent '{agent_name}' not found in agents.yaml")

    agent_cfg = agents[agent_name]

    # Resolve state_schema string -> class
    schema_path = agent_cfg.get("state_schema")
    if schema_path:
        agent_cfg["state_schema_cls"] = _resolve_state_schema(schema_path)

    return agent_cfg


# ---------------------------------------------------------------------------
# Final-node factory
# ---------------------------------------------------------------------------

def _create_llm_refine_final_node(refine_prompt_path: str, llm):
    """Final node that calls LLM with a refine prompt template.

    Used by detection agent to produce structured Chinese anomaly
    descriptions via LLM instead of string concatenation.
    """
    from langchain_core.messages import HumanMessage, AIMessage
    from app.utils.prompt_template import render_template
    from app.utils.llm_logger import get_trace_logger

    def final_node(state: Dict) -> Dict:
        logger.info("REGISTRY: Generating LLM-based final response")

        import json

        # Collect raw results for template
        three_sigma_raw = state.get("three_sigma_result")
        detection_raw_str = json.dumps(three_sigma_raw, ensure_ascii=False, default=str) if three_sigma_raw else "无"
        task_desc = state.get("task_description", "")

        # Build execution context so the refine LLM knows what happened
        context_parts = []
        csv_path = state.get("csv_file_path")
        if csv_path:
            context_parts.append(f"CSV 已加载: {csv_path}")
        tool_errors = state.get("tool_errors", [])
        if tool_errors:
            context_parts.append("执行错误:")
            for err in tool_errors:
                context_parts.append(f"  - {err.get('tool', '?')}: {err.get('error', '?')}")
        tool_results = state.get("tool_results", {})
        called_tools = list(tool_results.keys())
        if called_tools:
            context_parts.append(f"已调用工具: {', '.join(called_tools)}")
        else:
            context_parts.append("未调用任何工具")
        exec_context = "\n".join(context_parts)

        # Render template
        try:
            prompt = render_template(refine_prompt_path, {
                "DETECTION_RAW_DATA": detection_raw_str,
                "TASK_DESCRIPTION": task_desc,
                "EXECUTION_CONTEXT": exec_context,
            })
        except FileNotFoundError:
            logger.warning(f"Refine prompt not found: {refine_prompt_path}, using fallback")
            prompt = (
                f"你是一个 AIOps 异常检测专家。基于以下原始检测结果，用中文生成结构化的异常描述摘要。\n\n"
                f"## 任务描述\n{task_desc}\n\n"
                f"## 原始检测数据\n{detection_raw_str}\n\n"
                f"请按指标列出异常描述，包含 z-score、异常模式等信息。"
            )

        try:
            # M1: LLM retry + circuit breaker
            from app.middleware.llm_error_handling import get_llm_retry_handler
            result = get_llm_retry_handler().invoke(llm, [HumanMessage(content=prompt)])
            content = result.content if hasattr(result, "content") else str(result)

            # Trace LLM call
            get_trace_logger().log_llm_call(
                agent="detection_refine",
                input_messages=[HumanMessage(content=prompt)],
                response=AIMessage(content=content),
                metadata={"type": "detection_final_refinement"},
            )

            state["final_response"] = content

        except Exception as e:
            logger.error(f"REGISTRY: Detection refine LLM call failed: {e}")
            # Fallback: build a simple string summary
            state["final_response"] = _fallback_detection_summary(state)

        # Ensure abnormal_kpi is set for downstream consumers
        _ensure_abnormal_kpi(state)

        logger.info(f"REGISTRY: Final response generated ({len(state.get('final_response', ''))} chars)")
        return state

    return final_node


def _create_structured_final_node():
    """Final node that packages results as structured JSON (no LLM call).

    Used by generic agents — the supervisor synthesis handles report generation.
    """
    import json

    def final_node(state: Dict) -> Dict:
        logger.info("REGISTRY: Packaging structured results (no LLM)")
        parts = []

        csv_path = state.get("csv_file_path")
        inject_time = state.get("inject_time")
        abnormal_kpi = state.get("abnormal_kpi")
        rcd_result = state.get("rcd_result")
        pc_result = state.get("pc_result")
        graph_visualizations = state.get("graph_visualizations", [])
        tool_errors = state.get("tool_errors", [])

        if csv_path:
            parts.append(f"CSV 数据: {csv_path}")
        if inject_time:
            from app.utils.time_utils import format_unix_ts
            parts.append(f"注入时间: {format_unix_ts(inject_time)}")
        if abnormal_kpi:
            parts.append(f"异常指标: {abnormal_kpi}")

        if rcd_result and rcd_result.get("success"):
            parts.append(f"IAF-RCL 根因: {rcd_result.get('root_causes', [])[:20]}")
        if pc_result and pc_result.get("success"):
            parts.append(f"KE-FPC 根因: {pc_result.get('root_causes', [])[:20]}, 边: {len(pc_result.get('edges', []))}")

        if tool_errors:
            for err in tool_errors:
                parts.append(f"错误[{err.get('tool', '?')}]: {err.get('error', '?')}")

        state["final_response"] = "\n".join(parts)
        state["integrated_result"] = json.dumps({
            "rcd_result": rcd_result,
            "pc_result": pc_result,
            "csv_file_path": csv_path,
            "inject_time": inject_time,
            "abnormal_kpi": abnormal_kpi,
            "graph_visualizations": graph_visualizations,
            "tool_errors": tool_errors,
        }, ensure_ascii=False, default=str)

        logger.info("REGISTRY: Structured results packaged")
        return state

    return final_node


def _create_detection_structured_final_node():
    """Final node that transforms three_sigma_result into structured anomaly_report.

    No LLM call — descriptions are programmatically generated in Chinese.
    Sets ``anomaly_report`` as structured per-metric records and
    ``final_response`` as a simple summary string.
    """
    from datetime import datetime, timezone

    def final_node(state: Dict) -> Dict:
        logger.info("REGISTRY: Building structured detection output")

        three_sigma = state.get("three_sigma_result")
        anomaly_report = []

        if three_sigma and three_sigma.get("success"):
            anomalies_by_metric = three_sigma.get("anomalies_by_metric", {})
            parameters = three_sigma.get("parameters", {})
            metrics_checked = three_sigma.get("metrics_checked", 0)
            anomalies_found = three_sigma.get("anomalies_found", 0)

            for metric, info in anomalies_by_metric.items():
                points = info.get("points", [])
                if not points:
                    continue

                points_sorted = sorted(points, key=lambda p: p.get("timestamp", 0))

                time_start = points_sorted[0]["timestamp"]
                time_end = points_sorted[-1]["timestamp"]
                time_start_str = datetime.fromtimestamp(time_start, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                time_end_str = datetime.fromtimestamp(time_end, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

                z_scores = [p.get("z_score", 0) for p in points_sorted]
                values = [p.get("value", 0) for p in points_sorted]
                baseline_mean = points_sorted[0].get("baseline_mean", 0)
                baseline_std = points_sorted[0].get("baseline_std", 0)

                anomaly_type = info.get("anomaly_type", "sudden_increase")
                max_z = info.get("max_z_score", 0)
                type_cn = "突增" if anomaly_type == "sudden_increase" else "骤降"

                key_evidence = (
                    f"z-score 峰值 {max_z:.2f}，"
                    f"观测值从基线 {baseline_mean:.1f}±{baseline_std:.1f} "
                    f"飙升至 {min(values):.1f}~{max(values):.1f}"
                )

                description = (
                    f"指标 {metric} 在 {time_start_str} 至 {time_end_str} 期间发生{type_cn}，"
                    f"观测值从基线 {baseline_mean:.1f}±{baseline_std:.1f} "
                    f"偏离至 {min(values):.1f}~{max(values):.1f}，"
                    f"最大 z-score 为 {max_z:.2f}，"
                    f"共有 {len(points_sorted)} 个数据点超过 3σ 阈值。"
                )

                record = {
                    "metric": metric,
                    "anomaly_type": anomaly_type,
                    "anomaly_time_start": time_start,
                    "anomaly_time_end": time_end,
                    "anomaly_time_start_str": time_start_str,
                    "anomaly_time_end_str": time_end_str,
                    "max_z_score": max_z,
                    "anomaly_point_count": len(points_sorted),
                    "baseline_mean": baseline_mean,
                    "baseline_std": baseline_std,
                    "observation_min": min(values),
                    "observation_max": max(values),
                    "z_score_range": [round(min(z_scores), 2), round(max(z_scores), 2)],
                    "key_evidence": key_evidence,
                    "description": description,
                }
                anomaly_report.append(record)

            # Sort by max_z_score descending
            anomaly_report.sort(key=lambda r: r["max_z_score"], reverse=True)

            state["anomaly_report"] = anomaly_report

            # Store detection parameters for downstream report generation
            state["detection_parameters"] = {
                "algorithm": "3-sigma",
                "baseline_window_minutes": f"{parameters.get('baseline_start_minutes', 30)}~{parameters.get('baseline_end_minutes', 60)}",
                "detection_window_minutes": f"-{parameters.get('detect_before_minutes', 10)}~+{parameters.get('detect_minutes', 10)}",
                "threshold_sigma": parameters.get("threshold", 3.0),
                "metrics_checked": metrics_checked,
                "anomalous_metric_count": len(anomaly_report),
                "anomaly_point_count": anomalies_found,
            }

            # Build simple summary string
            if anomaly_report:
                parts = [
                    "3-Sigma 异常检测完成",
                    f"- 扫描指标数: {metrics_checked}",
                    f"- 异常指标数: {len(anomaly_report)}",
                    f"- 异常数据点: {anomalies_found}",
                    "\n最异常指标:",
                ]
                for r in anomaly_report[:5]:
                    type_cn = "突增" if r["anomaly_type"] == "sudden_increase" else "骤降"
                    parts.append(f"  - {r['metric']}: {type_cn}, max z-score={r['max_z_score']:.2f}")
                state["final_response"] = "\n".join(parts)
            else:
                state["final_response"] = (
                    f"3-Sigma 异常检测完成: 扫描 {three_sigma.get('metrics_checked', 0)} 个指标，"
                    f"未发现超过阈值的异常指标。"
                )
        else:
            err = three_sigma.get("error", "未执行") if three_sigma else "未执行"
            state["anomaly_report"] = []
            state["detection_parameters"] = None
            state["final_response"] = f"3-Sigma 异常检测失败: {err}"

        _ensure_abnormal_kpi(state)

        logger.info(f"REGISTRY: Structured detection output ({len(anomaly_report)} metrics)")
        return state

    return final_node


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def _create_route_after_extract(termination_signal: Optional[str] = None):
    """Create a route_after_extract function with optional termination signal.

    If *termination_signal* names a state key that is present and non-None
    in state (regardless of success/failure), routes directly to ``"final"``
    instead of looping.  This prevents the LLM from endlessly retrying a
    tool that already returned a result.
    """
    def route_fn(state: Dict) -> str:
        # Check interrupt
        if state.get("interrupted"):
            return "interrupt"

        # Check termination signal — stop as soon as the tool has produced
        # ANY result (success or failure).  Retrying the same tool won't help.
        if termination_signal:
            signal_data = state.get(termination_signal)
            if signal_data is not None:
                logger.info(f"REGISTRY: Termination signal '{termination_signal}' present → final")
                return "final"

        # Default: check if we should continue the ReAct loop
        from langchain_core.messages import AIMessage
        messages = state.get("messages", [])
        last_ai = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                last_ai = msg
                break

        if last_ai and hasattr(last_ai, "tool_calls") and last_ai.tool_calls:
            iteration = state.get("iteration_count", 0)
            max_iter = state.get("max_iterations", 10)
            if iteration < max_iter:
                return "model"

        return "final"

    return route_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fallback_detection_summary(state: Dict) -> str:
    """Build a simple string summary when LLM refine fails."""
    import json
    three_sigma = state.get("three_sigma_result", {})
    parts = []

    csv_path = state.get("csv_file_path", "")
    inject_time = state.get("inject_time")

    if csv_path:
        parts.append(f"CSV 数据文件: {csv_path}")
    if inject_time:
        from app.utils.time_utils import format_inject_time
        parts.append(f"故障注入时间: {format_inject_time(inject_time)}")

    if three_sigma and three_sigma.get("success"):
        anomalies = three_sigma.get("anomalies", [])
        parts.append(f"\n3-Sigma 异常检测成功: 发现 {len(anomalies)} 个异常指标")
        for a in anomalies[:10]:
            parts.append(f"  - {a['metric']}: z={a['z_score']:.2f}")
    else:
        err = three_sigma.get("error", "未执行") if three_sigma else "未执行"
        parts.append(f"\n3-Sigma 异常检测失败: {err}")

    return "\n".join(parts)


def _ensure_abnormal_kpi(state: Dict):
    """Set abnormal_kpi from top anomaly if not already set."""
    if state.get("abnormal_kpi"):
        return
    three_sigma = state.get("three_sigma_result", {})
    if isinstance(three_sigma, dict) and three_sigma.get("success"):
        anomalies = three_sigma.get("anomalies", [])
        if anomalies:
            state["abnormal_kpi"] = anomalies[0]["metric"]


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_react_agent(config: Dict[str, Any]):
    """Build a compiled ReAct graph from *config*.

    Config keys used:
        state_schema_cls   — TypedDict class for StateGraph
        system_prompt      — path to .md system prompt
        tools              — list of tool names
        model.provider     — LLM provider string
        model.temperature  — LLM temperature
        max_iterations     — ReAct loop limit
        first_iteration_instruction — optional instruction for iteration 1
        refine_prompt      — path to refine .md (triggers LLM-based final node)
        termination_signal — state key to check for early termination
    """
    logger.info(f"REGISTRY: Building agent '{config.get('name')}' via registry")

    # --- tools ---
    tool_names = config.get("tools", [])
    from app.tools.langchain_tool_adapters import create_langchain_tools
    tools = create_langchain_tools(tool_names)

    # --- prompt ---
    system_prompt = load_prompt(config["system_prompt"])

    # --- LLM ---
    model_cfg = config.get("model", {})
    from app.config.model_config import get_llm
    llm = get_llm(
        provider=model_cfg.get("provider"),
        temperature=model_cfg.get("temperature", 0),
    )
    llm_with_tools = llm.bind_tools(tools)

    # --- schema ---
    schema_cls = config.get("state_schema_cls")
    if schema_cls is None:
        raise ValueError(f"state_schema_cls not resolved for agent '{config.get('name')}'")

    # --- final node ---
    refine_prompt = config.get("refine_prompt")
    if refine_prompt:
        # LLM-based final node (e.g., when explicitly configured)
        refine_llm = get_llm(
            provider=model_cfg.get("provider"),
            temperature=0.3,  # slightly creative for structured descriptions
        )
        final_node = _create_llm_refine_final_node(refine_prompt, refine_llm)
    elif schema_cls and hasattr(schema_cls, '__annotations__') and 'anomaly_report' in schema_cls.__annotations__:
        # Detection agent: generate structured anomaly_report (no LLM)
        final_node = _create_detection_structured_final_node()
    else:
        # Generic structured JSON packaging (no LLM)
        final_node = _create_structured_final_node()

    # --- routing ---
    termination_signal = config.get("termination_signal")
    route_after_extract = _create_route_after_extract(termination_signal)

    # --- build graph ---
    builder = StateGraph(schema_cls)

    first_instruction = config.get("first_iteration_instruction")
    builder.add_node("model", create_model_node(
        llm_with_tools, system_prompt,
        first_iteration_instruction=first_instruction,
    ))
    # M2: Safe tool node (wraps ToolNode with exception handling)
    from app.middleware.tool_error_handling import create_safe_tool_node
    builder.add_node("tools", create_safe_tool_node(tools))
    builder.add_node("extract_results", extract_results_node)
    builder.add_node("final", final_node)

    builder.set_entry_point("model")

    builder.add_conditional_edges("model", route_after_model, {
        "tools": "tools",
        "final": "final",
    })
    builder.add_edge("tools", "extract_results")
    builder.add_conditional_edges("extract_results", route_after_extract, {
        "model": "model",
        "interrupt": "final",
        "final": "final",
    })
    builder.add_edge("final", END)

    graph = builder.compile()
    logger.info(f"REGISTRY: Agent '{config.get('name')}' compiled")
    return graph
