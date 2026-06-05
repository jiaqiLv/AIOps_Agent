"""Report Agent — Generates natural-language report from structured data.

Takes structured anomaly_report from detection and fault_type/root_causes/
propagation_path from diagnose, detects what data is available, selects the
appropriate prompt template, and calls LLM to produce the right kind of
report:

- Detection only → anomaly detection report (report_detection.md)
- Detection + Diagnose → root cause analysis report (report_system.md)

Graph structure:
START → generate_report_node → END
"""

import json
from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from app.models.report_agent_state import ReportAgentState
from app.config.model_config import get_deepseek_llm
from app.utils.logger import get_logger
from app.utils.prompt_loader import load_prompt
from app.utils.prompt_template import (
    format_detection_structured,
    format_diagnose_structured,
    render_template,
)
from app.utils.llm_logger import get_trace_logger

logger = get_logger(__name__)


def _select_template(has_detection: bool, has_diagnose: bool) -> str:
    """Select the appropriate prompt template based on available data.

    Returns the path to the prompt .md file.
    """
    if has_detection and has_diagnose:
        # Full root cause analysis: detection + diagnose
        return "app/prompts/report_system.md"
    elif has_detection:
        # Anomaly detection only
        return "app/prompts/report_detection.md"
    elif has_diagnose:
        # Diagnose only (edge case — should be rare)
        return "app/prompts/report_system.md"
    else:
        return None  # will trigger fallback


def generate_report_node(state: ReportAgentState) -> ReportAgentState:
    """Generate the final natural-language report from structured data.

    Detects what data is available (detection-only, diagnose-only, or both)
    and selects the appropriate prompt template to generate the right
    kind of report (anomaly detection report vs. root cause analysis report).
    """
    logger.info("REPORT: Generating NL report from structured data")

    task_description = state.get("task_description", "")
    detection_anomaly_report = state.get("detection_anomaly_report", [])
    detection_parameters = state.get("detection_parameters")
    diagnose_root_causes = state.get("diagnose_root_causes", [])
    diagnose_fault_type = state.get("diagnose_fault_type")
    diagnose_propagation_path = state.get("diagnose_propagation_path", [])
    csv_file_path = state.get("csv_file_path")
    inject_time = state.get("inject_time")
    abnormal_kpi = state.get("abnormal_kpi")
    tool_errors = state.get("tool_errors", [])

    has_detection = bool(detection_anomaly_report)
    has_diagnose = bool(diagnose_root_causes)

    # Format structured data
    detection_str = format_detection_structured(detection_anomaly_report, detection_parameters)
    diagnose_str = format_diagnose_structured(
        diagnose_fault_type, diagnose_root_causes, diagnose_propagation_path
    )

    # Add metadata context
    meta_parts = []
    if csv_file_path:
        meta_parts.append(f"CSV 数据文件: {csv_file_path}")
    if inject_time:
        from app.utils.time_utils import format_inject_time
        meta_parts.append(f"故障注入时间: {format_inject_time(inject_time)}")
    if abnormal_kpi:
        meta_parts.append(f"异常指标: {abnormal_kpi}")
    if tool_errors:
        meta_parts.append("执行错误:")
        for err in tool_errors:
            meta_parts.append(f"  - {err.get('tool', '?')}: {err.get('error', '?')}")

    meta_str = "\n".join(meta_parts) if meta_parts else "无额外上下文"

    # Select and render the appropriate template
    template_path = _select_template(has_detection, has_diagnose)
    report_type = "root cause analysis" if has_diagnose else "anomaly detection"

    if template_path:
        try:
            prompt = render_template(template_path, {
                "TASK_DESCRIPTION": task_description or "未提供",
                "DETECTION_STRUCTURED_DATA": detection_str,
                "DIAGNOSE_STRUCTURED_DATA": diagnose_str,
                "META_CONTEXT": meta_str,
            })
            logger.info(f"REPORT: Using template {template_path} ({report_type})")
        except FileNotFoundError:
            logger.warning(f"REPORT: Template {template_path} not found, using fallback")
            template_path = None

    if template_path is None:
        # Fallback prompt
        if has_detection and has_diagnose:
            prompt = (
                f"你是一个 AIOps 根因分析专家。请基于以下结构化分析数据生成完整的中文根因分析报告。\n\n"
                f"## 任务描述\n{task_description}\n\n"
                f"## 上下文\n{meta_str}\n\n"
                f"{detection_str}\n\n"
                f"{diagnose_str}\n\n"
                f"请生成包含分析概述、异常检测结果、根因指标列表、故障类型判断、结论与建议的结构化报告。"
            )
        else:
            prompt = (
                f"你是一个 AIOps 异常检测专家。请基于以下检测数据生成中文异常检测报告（可能包含 3-Sigma 和/或 BLD Metric (ECOD) 算法结果）。\n\n"
                f"## 任务描述\n{task_description}\n\n"
                f"## 上下文\n{meta_str}\n\n"
                f"{detection_str}\n\n"
                f"注意：仅生成异常检测报告，包含分析概述、异常检测结果（指标列表、检测解读）和后续建议。"
                f"不要编造根因指标或故障类型。"
            )

    try:
        llm = get_deepseek_llm(temperature=0.3)
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content if hasattr(response, "content") else str(response)

        get_trace_logger().log_llm_call(
            agent="report_agent",
            input_messages=[HumanMessage(content=prompt)],
            response=AIMessage(content=content),
            metadata={
                "report_type": report_type,
                "template": template_path or "fallback",
                "detection_metrics": len(detection_anomaly_report),
                "diagnose_root_causes": len(diagnose_root_causes),
            },
        )

        state["final_response"] = content
        logger.info(f"REPORT: NL report generated ({len(content)} chars, type={report_type})")

    except Exception as e:
        logger.error(f"REPORT: LLM call failed: {e}")
        # Fallback: concatenate structured data directly
        fallback_parts = [
            f"# {'AIOps 根因分析报告' if has_diagnose else 'AIOps 异常检测报告'}\n",
            f"## 任务描述\n{task_description}\n",
            detection_str,
        ]
        if has_diagnose:
            fallback_parts.append(diagnose_str)
        fallback_parts.append("\n注: LLM报告生成失败，以上为结构化分析结果。")
        state["final_response"] = "\n\n".join(fallback_parts)

    return state


def build_report_agent():
    """Build the report agent graph.

    Simple graph: START → generate_report → END

    Returns:
        Compiled StateGraph
    """
    logger.info("Building report agent")

    builder = StateGraph(ReportAgentState)

    builder.add_node("generate_report", generate_report_node)

    builder.set_entry_point("generate_report")
    builder.add_edge("generate_report", END)

    graph = builder.compile()
    logger.info("Report agent compiled")
    return graph


# Lazy-loaded graph
from app.utils.lazy_graph import LazyGraph
graph = LazyGraph(build_report_agent)
report_agent = graph
