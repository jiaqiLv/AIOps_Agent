"""Prompt template rendering utilities.

Provides template rendering with {{VAR}} placeholder substitution and
structured formatting functions for detection and diagnose results.
"""

from typing import Dict, Any, List, Optional

from app.utils.prompt_loader import load_prompt
from app.utils.time_utils import format_inject_time
from app.utils.logger import get_logger

logger = get_logger(__name__)


def render_template(template_path: str, variables: Dict[str, str]) -> str:
    """Load a .md template and replace {{VAR}} placeholders."""
    template = load_prompt(template_path)
    for key, value in variables.items():
        template = template.replace(f"{{{{{key}}}}}", value)
    return template


def format_detection_summary(detection_result: Optional[Dict[str, Any]]) -> str:
    """Format detection agent results into a structured summary string.

    Handles both 3-Sigma and BLD Metric (ECOD) results.

    Args:
        detection_result: The detection_result dict from supervisor state.

    Returns:
        Formatted markdown string for template injection.
    """
    if not detection_result:
        return "未执行异常检测。"

    parts = []

    # If detection_result has a 'summary' key (from subagent tool), use it directly
    summary = detection_result.get("summary")
    if summary:
        parts.append(summary)
        return "\n".join(parts)

    # Otherwise format from structured fields
    csv_path = detection_result.get("csv_file_path", "")
    inject_time = detection_result.get("inject_time")

    if csv_path:
        parts.append(f"CSV 数据文件: {csv_path}")
    if inject_time:
        parts.append(f"故障注入时间: {format_inject_time(inject_time)}")

    # ── Format 3-Sigma results ──────────────────────────
    three_sigma = detection_result.get("three_sigma_result")
    if three_sigma and three_sigma.get("success"):
        parts.append("\n3-Sigma 异常检测: 成功")
        parts.extend(_format_anomaly_by_metric(three_sigma.get("anomalies_by_metric", {}), "z-score"))
    elif three_sigma:
        error_msg = three_sigma.get("error", "Unknown")
        parts.append(f"\n3-Sigma 异常检测: 失败 — {error_msg}")

    # ── Format BLD Metric results ───────────────────────
    bld_metric = detection_result.get("bld_metric_result")
    if bld_metric and bld_metric.get("success"):
        parts.append("\nBLD Metric (ECOD) 异常检测: 成功")
        parts.extend(_format_anomaly_by_metric(bld_metric.get("anomalies_by_metric", {}), "ECOD score"))
    elif bld_metric:
        error_msg = bld_metric.get("error", "Unknown")
        parts.append(f"\nBLD Metric (ECOD) 异常检测: 失败 — {error_msg}")

    # ── Common: abnormal KPI ────────────────────────────
    if not parts and not three_sigma and not bld_metric:
        return "异常检测无结果。"

    abnormal_kpi = detection_result.get("abnormal_kpi")
    if abnormal_kpi:
        parts.append(f"\n最异常指标 (abnormal_kpi): {abnormal_kpi}")

    return "\n".join(parts) if parts else "异常检测无结果。"


def _format_anomaly_by_metric(anomalies_by_metric: dict, score_label: str) -> list:
    """Format anomalies_by_metric dict into markdown list lines."""
    lines = []
    if not anomalies_by_metric:
        return lines
    lines.append(f"异常指标数: {len(anomalies_by_metric)}")
    lines.append("")
    for metric, info in anomalies_by_metric.items():
        anomaly_type = info.get("anomaly_type", "unknown")
        type_cn = "突增" if anomaly_type == "sudden_increase" else "骤降"
        max_score = info.get("max_z_score") or info.get("max_score", 0)
        point_count = len(info.get("points", []))
        lines.append(
            f"- **{metric}**: {type_cn}, 最大 {score_label}={max_score:.2f}, "
            f"异常点数={point_count}"
        )
    lines.append("")
    return lines


def format_diagnose_summary(
    diagnose_result: Optional[Dict[str, Any]],
    graph_visualizations: Optional[List[Dict[str, Any]]] = None,
    tool_errors: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Format diagnose agent results into a structured summary string.

    Args:
        diagnose_result: The diagnose_result dict from supervisor state.
        graph_visualizations: Optional list of graph visualization results.
        tool_errors: Optional list of tool error records.

    Returns:
        Formatted markdown string for template injection.
    """
    if not diagnose_result:
        return "未执行根因分析。"

    parts = []

    # Analysis parameters
    inject_time = diagnose_result.get("inject_time")
    abnormal_kpi = diagnose_result.get("abnormal_kpi")
    csv_path = diagnose_result.get("csv_file_path")

    if csv_path:
        parts.append(f"CSV 数据文件: {csv_path}")
    if inject_time:
        parts.append(f"故障注入时间: {format_inject_time(inject_time)}")
    if abnormal_kpi:
        parts.append(f"异常指标: {abnormal_kpi}")

    # RCD results
    rcd_result = diagnose_result.get("rcd_result")
    if rcd_result:
        parts.append("\n## IAF-RCL 算法结果")
        if rcd_result.get("success"):
            rc = rcd_result.get("root_causes", [])
            parts.append(f"- 状态: 成功\n- 根因数量: {len(rc)}")
            if rc:
                parts.append(f"- 根因列表 (前20): {rc[:20]}")
        else:
            parts.append(f"- 状态: 失败\n- 错误: {rcd_result.get('error', 'Unknown')}")

    # PC results
    pc_result = diagnose_result.get("pc_result")
    if pc_result:
        parts.append("\n## KE-FPC 算法结果")
        if pc_result.get("success"):
            rc = pc_result.get("root_causes", [])
            edges = pc_result.get("edges", [])
            parts.append(f"- 状态: 成功\n- 根因数量: {len(rc)}\n- 因果边: {len(edges)}")
            if rc:
                parts.append(f"- 根因列表 (前20): {rc[:20]}")
            if edges:
                parts.append(f"- 因果边 (前30): {edges[:30]}")
        else:
            parts.append(f"- 状态: 失败\n- 错误: {pc_result.get('error', 'Unknown')}")

    # Graph visualizations
    if graph_visualizations:
        parts.append("\n## 故障传播图可视化")
        for viz in graph_visualizations:
            filepath = viz.get("filepath")
            fmt = viz.get("format", "html")
            parts.append(f"- {fmt.upper()} 格式图: {filepath}")
            if viz.get("abnormal_kpi"):
                parts.append(f"  异常指标: {viz['abnormal_kpi']}")

    # Tool errors
    if tool_errors:
        parts.append("\n## 执行错误")
        for err in tool_errors:
            parts.append(f"- {err.get('tool', 'unknown')}: {err.get('error', 'unknown')}")

    # Final response from diagnose agent
    final_response = diagnose_result.get("summary") or diagnose_result.get("final_response")
    if final_response:
        parts.append(f"\n## 诊断摘要\n{final_response}")

    return "\n".join(parts) if parts else "根因分析无结果。"


# ==================== Structured Data Formatting ====================


def format_detection_structured(
    anomaly_report: Optional[List[Dict[str, Any]]],
    detection_parameters: Optional[Dict[str, Any]] = None,
) -> str:
    """Format detection anomaly_report records as markdown string.

    Handles both 3-Sigma and BLD Metric (ECOD) records.  Each record carries
    an ``algorithm`` field ("3-sigma" or "bld_metric_ecod") that determines
    which columns are rendered.

    Args:
        anomaly_report: List of per-metric anomaly records from detection agent.
        detection_parameters: Detection algorithm parameters.

    Returns:
        Formatted markdown string for template injection.
    """
    if not anomaly_report:
        return "异常检测未返回结果。"

    # Detect which algorithms contributed
    algorithms = set(r.get("algorithm", "3-sigma") for r in anomaly_report)
    algo_names = []
    if "3-sigma" in algorithms:
        algo_names.append("3-Sigma")
    if "bld_metric_ecod" in algorithms:
        algo_names.append("BLD Metric (ECOD)")
    algo_label = " + ".join(algo_names) if algo_names else "异常检测"

    parts = [f"## {algo_label} 异常检测结果\n"]

    # Build detection method description from parameters
    if detection_parameters:
        algo = detection_parameters.get("algorithm", "")
        method_parts = []
        if "3-sigma" in algo:
            baseline_win = detection_parameters.get("baseline_window_minutes", "?")
            detection_win = detection_parameters.get("detection_window_minutes", "?")
            threshold = detection_parameters.get("threshold_sigma", 3.0)
            method_parts.append(
                f"3-Sigma（基线窗口: {baseline_win} / 检测窗口: {detection_win} / 阈值: {threshold}σ）"
            )
        if "bld_metric" in algo:
            train_h = detection_parameters.get("train_hours", "?")
            contam = detection_parameters.get("contamination", "?")
            method_parts.append(
                f"BLD Metric (ECOD)（训练: 前 {train_h} h / contamination: {contam}）"
            )
        if method_parts:
            parts.append(f"**检测方法**: {' + '.join(method_parts)}\n")
        metrics_checked = detection_parameters.get("metrics_checked", 0)
        anomalous_count = detection_parameters.get("anomalous_metric_count", len(anomaly_report))
        anomaly_points = detection_parameters.get("anomaly_point_count", 0)
        parts.append(
            f"**检测概况**: 扫描 {metrics_checked} 个指标，"
            f"检出 {anomalous_count} 个异常指标，共 {anomaly_points} 个异常数据点\n"
        )
    else:
        parts.append(f"检出异常指标 {len(anomaly_report)} 个：\n")

    # ── Summary table ────────────────────────────────────
    parts.append("\n### 异常指标概览\n")
    has_bld = "bld_metric_ecod" in algorithms
    if has_bld:
        parts.append(
            "| # | 算法 | 指标名称 | 异常类型 | 异常时段 | 关键得分 | 基线/中位数 | 观测值范围 |"
        )
        parts.append(
            "|---|------|---------|---------|---------|---------|----------|-----------|"
        )
    else:
        parts.append(
            "| # | 指标名称 | 异常类型 | 异常时段 | z-score | 基线 μ±σ | 观测值范围 |"
        )
        parts.append(
            "|---|---------|---------|---------|---------|----------|-----------|"
        )

    for i, record in enumerate(anomaly_report, 1):
        anomaly_type = record.get("anomaly_type", "unknown")
        type_cn = "突增" if anomaly_type == "sudden_increase" else "骤降"
        time_start = record.get("anomaly_time_start_str", "?")
        time_end = record.get("anomaly_time_end_str", "?")
        if time_start != "?" and time_end != "?" and time_start[:10] == time_end[:10]:
            time_range = f"{time_start[11:]} ~ {time_end[11:]}"
        else:
            time_range = f"{time_start} ~ {time_end}"

        algo = record.get("algorithm", "3-sigma")
        if algo == "bld_metric_ecod":
            algo_short = "BLD Metric"
            score_str = f"score={record.get('max_score', 0):.4f}"
            baseline_str = f"median={record.get('training_median', 0):.1f}"
        else:
            algo_short = "3-Sigma"
            score_str = f"z={record.get('max_z_score', 0):.2f}"
            baseline_str = (
                f"μ={record.get('baseline_mean', 0):.1f} "
                f"σ={record.get('baseline_std', 0):.1f}"
            )

        if has_bld:
            parts.append(
                f"| {i} | {algo_short} | {record['metric']} | {type_cn} | {time_range} | "
                f"{score_str} | {baseline_str} | "
                f"{record.get('observation_min', 0):.1f} ~ {record.get('observation_max', 0):.1f} |"
            )
        else:
            parts.append(
                f"| {i} | {record['metric']} | {type_cn} | {time_range} | "
                f"{record.get('max_z_score', 0):.2f} | "
                f"{record.get('baseline_mean', 0):.1f}±{record.get('baseline_std', 0):.1f} | "
                f"{record.get('observation_min', 0):.1f} ~ {record.get('observation_max', 0):.1f} |"
            )

    # --- Detail section ---
    parts.append("\n### 异常事件详述\n")

    for i, record in enumerate(anomaly_report, 1):
        anomaly_type = record.get("anomaly_type", "unknown")
        type_cn = "突增" if anomaly_type == "sudden_increase" else "骤降"
        parts.append(
            f"**{i}. {record['metric']}** — {type_cn}\n"
            f"   - **关键证据**: {record.get('key_evidence', '无')}\n"
            f"   - **异常描述**: {record.get('description', '无')}\n"
        )

    return "\n".join(parts)


def format_diagnose_structured(
    fault_type: Optional[str],
    root_causes: Optional[List[Dict[str, Any]]],
    propagation_path: Optional[List[Dict[str, Any]]],
) -> str:
    """Format diagnose structured data as markdown string.

    Produces:
      1. Fault type
      2. Root cause summary table (overview)
      3. Root cause detail section (per-metric reasoning)
      4. Propagation path (inline, for LLM understanding only)

    Args:
        fault_type: Inferred fault type string.
        root_causes: List of root cause records with metric, confidence, etc.
        propagation_path: List of propagation edges.

    Returns:
        Formatted markdown string for template injection.
    """
    parts = ["## 根因分析结果\n"]

    # Fault type
    if fault_type:
        parts.append(f"**故障类型**: {fault_type}\n")

    # Root causes
    if root_causes:
        # --- Summary table ---
        level_cn = {"high": "高", "medium": "中", "low": "低", "unknown": "未知"}
        parts.append(f"### 根因概览 ({len(root_causes)} 个)\n")
        parts.append("| # | 指标名称 | 置信度 | 级别 | 支持算法 |")
        parts.append("|---|---------|--------|------|---------|")
        for i, rc in enumerate(root_causes, 1):
            confidence_level = rc.get("confidence_level", "unknown")
            algorithms = " + ".join(rc.get("supporting_algorithms", []))
            parts.append(
                f"| {i} | {rc['metric']} | {rc.get('confidence', 0):.2f} | "
                f"{level_cn.get(confidence_level, confidence_level)} | {algorithms} |"
            )

        # --- Detail section ---
        parts.append("\n### 根因推理依据\n")
        for i, rc in enumerate(root_causes, 1):
            confidence_level = rc.get("confidence_level", "unknown")
            algorithms = " + ".join(rc.get("supporting_algorithms", []))
            level_label = level_cn.get(confidence_level, confidence_level)
            parts.append(
                f"{i}. **{rc['metric']}** (置信度: {rc.get('confidence', 0):.2f}, "
                f"{level_label}, {algorithms})\n"
                f"   - 入选理由: {rc.get('reason', '无')}\n"
            )
    else:
        parts.append("未识别出根因指标。\n")

    # Propagation path (for LLM understanding, not for report output)
    if propagation_path:
        parts.append(f"\n### 故障传播路径 ({len(propagation_path)} 条边)\n")
        for edge in propagation_path:
            weight = edge.get("weight", 0)
            parts.append(
                f"- {edge.get('source', '?')} → {edge.get('target', '?')} "
                f"(权重: {weight:.3f})\n"
            )

    return "\n".join(parts) if len(parts) > 1 else "根因分析未返回结果。"
