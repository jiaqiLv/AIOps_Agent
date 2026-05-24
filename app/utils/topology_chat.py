"""Format propagation topology for LangGraph Studio chat (markdown-safe)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage

from app.config.algorithm_names import ALGORITHM_KE_FPC
from app.tools.graph_visualization_tool import (
    get_topology_embed_url,
    get_topology_view_url,
)


def metric_semantic_label(metric: str) -> str:
    name = metric.lower()
    if "error" in name and "rate" in name:
        return "错误率上升"
    if "cpu" in name:
        return "CPU资源饱和"
    if "memory" in name or "_mem" in name:
        return "内存压力升高"
    if "duration" in name or "latency" in name or "_rt" in name or "response" in name:
        return "响应时间飙升"
    if "count" in name or "qps" in name or "throughput" in name:
        return "请求量激增"
    if "disk" in name or "iowait" in name:
        return "磁盘IO瓶颈"
    if "network" in name or "packet" in name:
        return "网络异常"
    if "connection" in name or "pool" in name:
        return "连接资源紧张"
    return "指标异常"


def format_propagation_path_line(index: int, path: List[str]) -> str:
    if not path:
        return ""
    segments = [f"{node} ({metric_semantic_label(node)})" for node in path]
    return f"路径{index}: " + " → ".join(segments)


def format_propagation_paths_section(
    paths: List[List[str]],
    *,
    title: str | None = None,
) -> str:
    if not paths:
        return ""
    heading = title or f"## 基于 {ALGORITHM_KE_FPC} 的因果图：故障传播路径"
    lines = [heading, ""]
    for i, path in enumerate(paths, 1):
        line = format_propagation_path_line(i, path)
        if line:
            lines.append(line)
    return "\n".join(lines)


def paths_to_mermaid_flowchart(
    paths: List[List[str]],
    edges: Optional[List[List[str]]] = None,
) -> str:
    edge_set: set[tuple[str, str]] = set()
    for path in paths:
        for i in range(len(path) - 1):
            edge_set.add((path[i], path[i + 1]))
    if edges:
        for e in edges:
            if len(e) >= 2:
                edge_set.add((e[0], e[1]))
    if not edge_set:
        return ""

    def _nid(name: str) -> str:
        return "n_" + "".join(c if c.isalnum() else "_" for c in name)[:48]

    id_map = {n: _nid(n) for n in {n for pair in edge_set for n in pair}}
    lines = ["```mermaid", "flowchart LR"]
    for source, target in sorted(edge_set):
        sid, tid = id_map[source], id_map[target]
        sl = source.replace('"', "'")[:36]
        tl = target.replace('"', "'")[:36]
        lines.append(f'  {sid}["{sl}"] --> {tid}["{tl}"]')
    lines.append("```")
    return "\n".join(lines)


def build_topology_markdown_block(graph_viz: Dict[str, Any]) -> str:
    """Studio-friendly block: paths + mermaid + interactive topology links."""
    if not graph_viz or not graph_viz.get("success"):
        return ""

    paths = graph_viz.get("propagation_paths") or []
    edges = graph_viz.get("edges") or []
    parts: List[str] = [
        "## 根因传播拓扑（KE-FPC）",
        "",
    ]

    path_text = format_propagation_paths_section(paths)
    if path_text:
        parts.append(path_text)
        parts.append("")

    mermaid = paths_to_mermaid_flowchart(paths, edges)
    if mermaid:
        parts.extend(["### 传播拓扑流程图", "", mermaid, ""])

    view_url = graph_viz.get("view_url") or get_topology_view_url()
    embed_url = graph_viz.get("embed_url") or get_topology_embed_url()
    parts.extend([
        "### 可交互传播拓扑",
        "",
        f"- **完整页面**: [{view_url}]({view_url})",
        f"- **可拖拽交互页**: [{embed_url}]({embed_url})",
        "",
        "> 在浏览器中打开后：**用鼠标拖动节点**即可调整位置，节点文字已放大便于阅读。",
    ])

    return "\n".join(parts)


def build_final_report_messages(
    report_text: str,
    graph_viz: Optional[Dict[str, Any]],
) -> List[AIMessage]:
    """Two chat bubbles: (1) topology markdown (2) LLM analysis — Studio renders both."""
    messages: List[AIMessage] = []
    topology_block = build_topology_markdown_block(graph_viz) if graph_viz else ""

    if topology_block:
        messages.append(AIMessage(content=topology_block))

    analysis = (report_text or "").strip() or "根因分析已完成。"
    if not analysis.startswith("#"):
        analysis = "## 根因分析详情\n\n" + analysis
    messages.append(AIMessage(content=analysis))

    return messages


def build_final_report_message(
    report_text: str,
    graph_viz: Optional[Dict[str, Any]],
) -> AIMessage:
    """Single combined message fallback."""
    topology_block = build_topology_markdown_block(graph_viz) if graph_viz else ""
    analysis = (report_text or "").strip()
    if topology_block:
        body = topology_block + "\n\n---\n\n" + (analysis or "")
    else:
        body = analysis or "根因分析已完成。"
    return AIMessage(content=body)
