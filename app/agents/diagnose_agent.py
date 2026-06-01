"""Diagnose Agent - ReAct loop for root cause analysis

Uses a ReAct loop where the LLM autonomously decides which tools to call:
1. csv_reader_tool — load CSV metrics data
2. rcd_tool — IAF-RCL algorithm (requires inject_time)
3. pc_tool — KE-FPC causal discovery algorithm
4. graph_visualization_tool — generate propagation topology HTML

The diagnose agent produces structured output (fault_type, root_causes,
propagation_path) via programmatic extraction from algorithm results.
No LLM call in the final node.

Graph structure:
START → model(LLM+bind_tools) ──有tool_calls──→ ToolNode → extract_results → model(循环)
              │                                                              ↑
              └──无tool_calls──→ final(structured output) → END              │
"""

import json
from typing import Dict, Any, List, Optional
from collections import Counter
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from app.models.react_agent_state import ReactAgentState
from app.config.model_config import get_deepseek_llm
from app.tools.langchain_tool_adapters import create_diagnose_tools
from app.agents.nodes.react_nodes import (
    create_model_node,
    extract_results_node,
    route_after_model,
    route_after_extract,
)
from app.utils.logger import get_logger
from app.utils.prompt_loader import load_prompt

logger = get_logger(__name__)


# Fault type inference mapping
_FAULT_TYPE_MAP = {
    "cpu_usage": "CPU_RESOURCE_SATURATION",
    "cpu_util": "CPU_RESOURCE_SATURATION",
    "cpu": "CPU_RESOURCE_SATURATION",
    "memory": "MEMORY_LEAK",
    "mem": "MEMORY_LEAK",
    "mem_usage": "MEMORY_LEAK",
    "latency": "LATENCY_SPIKE",
    "rt": "LATENCY_SPIKE",
    "duration": "LATENCY_SPIKE",
    "error_rate": "ERROR_SURGE",
    "errors": "ERROR_SURGE",
    "error": "ERROR_SURGE",
    "disk": "DISK_BOTTLENECK",
    "iowait": "DISK_BOTTLENECK",
    "network": "NETWORK_ANOMALY",
    "packet": "NETWORK_ANOMALY",
    "db": "DATABASE_BOTTLENECK",
    "query": "DATABASE_BOTTLENECK",
    "connection": "DATABASE_BOTTLENECK",
}


def _infer_fault_type(metric_names: List[str]) -> str:
    """Infer fault type from root cause metric names via mapping table."""
    types = []
    for name in metric_names:
        name_lower = name.lower()
        for key, ftype in _FAULT_TYPE_MAP.items():
            if key in name_lower:
                types.append(ftype)
                break
    if not types:
        return "UNKNOWN"
    return Counter(types).most_common(1)[0][0]


def _compute_confidence(metric: str, rcd_root_causes: Optional[List], pc_root_causes: Optional[List]) -> tuple:
    """Compute confidence score/level/algorithms/reason for a root cause metric.

    Rules:
        - Both algorithms agree → high (0.85)
        - Only IAF-RCL → medium (0.6)
        - Only KE-FPC → low (0.4)
    """
    in_rcd = metric in (rcd_root_causes or [])
    in_pc = metric in (pc_root_causes or [])
    if in_rcd and in_pc:
        return 0.85, "high", ["IAF-RCL", "KE-FPC"], "IAF-RCL 和 KE-FPC 均识别为根因，因果路径明确"
    elif in_rcd:
        return 0.6, "medium", ["IAF-RCL"], "仅 IAF-RCL 识别为根因"
    elif in_pc:
        return 0.4, "low", ["KE-FPC"], "仅 KE-FPC 识别为根因"
    return 0.0, "unknown", [], ""


def _create_diagnose_structured_final_node():
    """Create final node that builds structured diagnose output (no LLM call).

    Extracts fault_type, root_causes, and propagation_path from rcd_result
    and pc_result. Generates programmatic descriptions only.
    Final response is a simple structured summary string.
    """
    def final_response_node(state: ReactAgentState) -> ReactAgentState:
        """Build structured diagnose output from algorithm results."""
        logger.info("DIAGNOSE: Building structured output (no LLM)")

        rcd_result = state.get("rcd_result")
        pc_result = state.get("pc_result")
        csv_file_path = state.get("csv_file_path")
        inject_time = state.get("inject_time")
        abnormal_kpi = state.get("abnormal_kpi")
        graph_visualizations = state.get("graph_visualizations", [])
        tool_errors = state.get("tool_errors", [])

        # ---- Extract raw root cause lists ----
        rcd_root_causes = (rcd_result or {}).get("root_causes", []) or []
        pc_root_causes = (pc_result or {}).get("root_causes", []) or []
        pc_edges = (pc_result or {}).get("edges", []) or []

        # ---- Build combined root_causes with confidence ----
        all_metrics = set()
        for rc in rcd_root_causes:
            if isinstance(rc, dict):
                all_metrics.add(rc.get("metric", rc.get("name", str(rc))))
            elif isinstance(rc, str):
                all_metrics.add(rc)
        for rc in pc_root_causes:
            if isinstance(rc, dict):
                all_metrics.add(rc.get("metric", rc.get("name", str(rc))))
            elif isinstance(rc, str):
                all_metrics.add(rc)

        structured_root_causes = []
        for metric in all_metrics:
            confidence, level, algorithms, reason = _compute_confidence(metric, rcd_root_causes, pc_root_causes)
            if confidence > 0:
                structured_root_causes.append({
                    "metric": metric,
                    "confidence": confidence,
                    "confidence_level": level,
                    "supporting_algorithms": algorithms,
                    "reason": reason,
                })

        # Sort by confidence descending
        structured_root_causes.sort(key=lambda rc: rc["confidence"], reverse=True)

        # ---- Infer fault type ----
        fault_type = _infer_fault_type(list(all_metrics)) if all_metrics else None

        # ---- Build propagation_path ----
        propagation_path = []
        for edge in pc_edges:
            if isinstance(edge, dict):
                propagation_path.append({
                    "source": edge.get("source", edge.get("from", "?")),
                    "target": edge.get("target", edge.get("to", "?")),
                    "weight": edge.get("weight", edge.get("strength", 0)),
                })
            elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                propagation_path.append({
                    "source": str(edge[0]),
                    "target": str(edge[1]),
                    "weight": edge[2] if len(edge) > 2 else 0,
                })

        state["fault_type"] = fault_type
        state["root_causes"] = structured_root_causes
        state["propagation_path"] = propagation_path

        # ---- Build programmatic final_response ----
        parts = []
        if csv_file_path:
            parts.append(f"CSV 数据: {csv_file_path}")
        from app.utils.time_utils import format_inject_time
        if inject_time:
            parts.append(f"注入时间: {format_inject_time(inject_time)}")
        if abnormal_kpi:
            parts.append(f"异常指标: {abnormal_kpi}")

        if structured_root_causes:
            parts.append(f"\n根因指标 ({len(structured_root_causes)} 个):")
            for rc in structured_root_causes:
                parts.append(f"  - {rc['metric']} (置信度: {rc['confidence']:.2f}, {rc['confidence_level']})")
        if fault_type:
            parts.append(f"\n故障类型: {fault_type}")
        if propagation_path:
            parts.append(f"\n传播路径: {len(propagation_path)} 条边")
            for pp in propagation_path[:10]:
                parts.append(f"  - {pp['source']} → {pp['target']} ({pp['weight']:.3f})")

        if tool_errors:
            parts.append("\n错误:")
            for err in tool_errors:
                parts.append(f"  - {err.get('tool', '?')}: {err.get('error', '?')}")

        state["final_response"] = "\n".join(parts) if parts else "根因分析未产生结果。"

        # ---- Preserve integrated_result for backward compat ----
        state["integrated_result"] = json.dumps({
            "rcd_result": rcd_result,
            "pc_result": pc_result,
            "csv_file_path": csv_file_path,
            "inject_time": inject_time,
            "abnormal_kpi": abnormal_kpi,
            "fault_type": fault_type,
            "root_causes": structured_root_causes,
            "propagation_path": propagation_path,
            "graph_visualizations": graph_visualizations,
            "tool_errors": tool_errors,
        }, ensure_ascii=False, default=str)

        logger.info(
            f"DIAGNOSE: Structured output: fault_type={fault_type}, "
            f"root_causes={len(structured_root_causes)}, edges={len(propagation_path)}"
        )
        # Exclude messages from return — prevents add_messages duplication
        result = {k: v for k, v in state.items() if k != "messages"}
        return result

    return final_response_node


def build_diagnose_agent():
    """Build the diagnose agent with ReAct loop.

    The LLM autonomously decides which analysis tools to call based on
    available parameters and intermediate results.

    Returns:
        Compiled StateGraph for diagnose agent
    """
    logger.info("Building diagnose agent with ReAct loop")

    # Load tools and prompt
    tools = create_diagnose_tools()
    system_prompt = load_prompt("app/prompts/diagnose_system.md")

    # Create LLM with bound tools
    llm = get_deepseek_llm(temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    # Build graph
    builder = StateGraph(ReactAgentState)

    # Add nodes
    builder.add_node("model", create_model_node(
        llm_with_tools, system_prompt,
        first_iteration_instruction="请立即调用 csv_reader_tool 开始分析数据。",
    ))
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("extract_results", extract_results_node)
    builder.add_node("final", _create_diagnose_structured_final_node())

    # Set entry point
    builder.set_entry_point("model")

    # Add edges
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
    logger.info("Diagnose agent compiled with ReAct loop")
    return graph


# Lazy-loaded graph (built on first access, not at import time)
from app.utils.lazy_graph import LazyGraph
graph = LazyGraph(build_diagnose_agent)
diagnose_agent = graph
