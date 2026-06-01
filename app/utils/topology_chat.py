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


# ==================== Self-contained HTML Report ====================

import base64
import os
import shutil
import time
from pathlib import Path

from app.utils.logger import get_logger
from app.utils.prompt_template import format_detection_summary

logger = get_logger(__name__)

_REPORTS_DIR = Path(os.getenv("REPORT_OUTPUT_DIR", "outputs/reports"))


def _resolve_reports_dir() -> Path:
    """Absolute path to reports output directory."""
    project_root = Path(__file__).resolve().parent.parent.parent
    d = _REPORTS_DIR if _REPORTS_DIR.is_absolute() else project_root / _REPORTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _markdown_to_simple_html(md: str) -> str:
    """Best-effort markdown -> HTML for report body (headings, bold, lists, code)."""
    lines = md.split("\n")
    html_parts: List[str] = []
    in_list = False
    in_code_block = False
    code_buf: List[str] = []

    for line in lines:
        if line.strip().startswith("```"):
            if in_code_block:
                html_parts.append(f"<pre><code>{_escape_html(chr(10).join(code_buf))}</code></pre>")
                code_buf = []
                in_code_block = False
            else:
                if in_list:
                    html_parts.append("</ul>")
                    in_list = False
                in_code_block = True
            continue
        if in_code_block:
            code_buf.append(line)
            continue

        stripped = line.strip()
        # Close list on non-list line
        if in_list and not stripped.startswith("- ") and not stripped.startswith("* ") and not stripped.startswith(tuple(f"{i}. " for i in range(1, 10))):
            html_parts.append("</ul>")
            in_list = False

        if stripped.startswith("### "):
            html_parts.append(f"<h3>{stripped[4:]}</h3>")
        elif stripped.startswith("## "):
            html_parts.append(f"<h2>{stripped[3:]}</h2>")
        elif stripped.startswith("# "):
            html_parts.append(f"<h1>{stripped[2:]}</h1>")
        elif stripped.startswith("- ") or stripped.startswith("* "):
            html_parts.append(f"<li>{stripped[2:]}</li>")
            if not in_list:
                html_parts.append("<ul>")
                in_list = True
        elif stripped.startswith(tuple(f"{i}. " for i in range(1, 10))):
            content = stripped[3:]
            html_parts.append(f"<li>{content}</li>")
            if not in_list:
                html_parts.append("<ol>")
                in_list = True
        elif stripped.startswith("---"):
            html_parts.append("<hr/>")
        elif stripped:
            html_parts.append(f"<p>{stripped}</p>")

    if in_list:
        html_parts.append("</ul>")
    if in_code_block:
        html_parts.append(f"<pre><code>{_escape_html(chr(10).join(code_buf))}</code></pre>")

    return "\n".join(html_parts)


def generate_html_report(
    report_text: str,
    graph_viz: Optional[Dict[str, Any]],
    detection_result: Optional[Dict[str, Any]] = None,
    task_description: str = "",
) -> str:
    """Generate a self-contained HTML report with embedded images.

    Returns the absolute path to the generated HTML file.
    """
    reports_dir = _resolve_reports_dir()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"report_{timestamp}.html"
    filepath = reports_dir / filename

    # ----- Build topology section -----
    topology_section = ""
    if graph_viz and graph_viz.get("success"):
        base64_png = graph_viz.get("png_base64")
        paths = graph_viz.get("propagation_paths") or []
        root_causes = graph_viz.get("root_causes") or []
        abnormal_kpi = graph_viz.get("abnormal_kpi") or "未指定"

        parts: List[str] = []

        if base64_png:
            parts.append(
                f'<img src="data:image/png;base64,{base64_png}" '
                f'alt="传播拓扑图" style="max-width:100%;border-radius:8px;'
                f'border:1px solid #e2e8f0;box-shadow:0 2px 12px rgba(0,0,0,0.08);"/>'
            )

        if paths:
            parts.append("<h3>故障传播路径</h3><ol>")
            for i, path in enumerate(paths[:15], 1):
                line = format_propagation_path_line(i, path)
                if line:
                    parts.append(f"<li>{_escape_html(line)}</li>")
            if len(paths) > 15:
                parts.append(f"<li>… 另有 {len(paths) - 15} 条路径</li>")
            parts.append("</ol>")

        if root_causes:
            parts.append("<h3>根因指标</h3><ul>")
            for rc in root_causes[:20]:
                parts.append(f"<li><strong>{_escape_html(rc)}</strong></li>")
            parts.append("</ul>")

        parts.append(f"<p><strong>异常 KPI:</strong> {_escape_html(abnormal_kpi)}</p>")

        topology_section = "\n".join(parts)

    # ----- Build detection section -----
    detection_section = ""
    if detection_result and detection_result.get("success"):
        detection_section = format_detection_summary(detection_result)
        detection_section = _markdown_to_simple_html(detection_section)

    # ----- Build report body -----
    report_html = _markdown_to_simple_html(report_text or "根因分析已完成。")

    # ----- Assemble full HTML -----
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>根因分析报告 - {timestamp}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Microsoft YaHei','PingFang SC','Segoe UI',Arial,sans-serif;
    background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
    color: #1e293b; line-height: 1.7; padding: 24px;
  }}
  .container {{ max-width: 960px; margin: 0 auto; }}
  h1 {{ font-size: 1.8rem; color: #0f172a; margin-bottom: 8px; }}
  h2 {{ font-size: 1.4rem; color: #1e40af; margin: 24px 0 12px;
       border-bottom: 2px solid #bfdbfe; padding-bottom: 6px; }}
  h3 {{ font-size: 1.15rem; color: #334155; margin: 16px 0 8px; }}
  p  {{ margin: 8px 0; }}
  ul, ol {{ margin: 8px 0 8px 24px; }}
  li {{ margin: 4px 0; }}
  hr {{ border: none; border-top: 1px solid #e2e8f0; margin: 20px 0; }}
  pre {{
    background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 6px;
    padding: 12px 16px; overflow-x: auto; margin: 12px 0;
  }}
  code {{ font-family: 'Cascadia Code','Fira Code',monospace; font-size: 0.9rem; }}
  .meta {{ color: #64748b; font-size: 0.9rem; margin-bottom: 20px; }}
  .section {{
    background: #ffffff; border-radius: 12px; padding: 24px 28px;
    border: 1px solid #e2e8f0; margin-bottom: 20px;
    box-shadow: 0 1px 8px rgba(0,0,0,0.04);
  }}
  .topology-img {{ text-align: center; margin: 16px 0; }}
</style>
</head>
<body>
<div class="container">
  <h1>根因分析报告</h1>
  <div class="meta">生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}"
  {f" | 任务: {_escape_html(task_description[:120])}" if task_description else ""}
  </div>

  {"<div class='section'>" + topology_section + "</div>" if topology_section else ""}

  {"<div class='section'><h2>异常检测结果</h2>" + detection_section + "</div>" if detection_section else ""}

  <div class="section">
    <h2>分析报告</h2>
    {report_html}
  </div>
</div>
</body>
</html>"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)

    # Copy as latest
    latest_path = reports_dir / "report_latest.html"
    shutil.copy2(filepath, latest_path)

    logger.info(f"Generated HTML report: {filepath} (latest: {latest_path})")
    return str(filepath)
