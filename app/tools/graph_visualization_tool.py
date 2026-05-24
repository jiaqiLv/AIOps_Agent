"""Fault Propagation Graph Visualization Tool

This tool generates visualizations of fault propagation graphs from KE-FPC algorithm results,
with different colors for KPI nodes and root cause nodes.
"""

import os
import base64
import shutil
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import json

from app.utils.logger import get_logger
from app.utils.propagation_paths import (
    filter_propagation_chain_edges,
    list_propagation_paths,
    orient_propagation_edges,
)

LATEST_GRAPH_FILENAME = "propagation_graph_latest.html"

logger = get_logger(__name__)


def get_topology_view_url() -> str:
    """Public URL for the latest propagation topology page."""
    base = os.getenv("LANGGRAPH_PUBLIC_BASE_URL", "http://127.0.0.1:2024").rstrip("/")
    return f"{base}/topology/latest"


def get_topology_embed_url() -> str:
    """Fullscreen draggable topology (recommended for interaction)."""
    base = os.getenv("LANGGRAPH_PUBLIC_BASE_URL", "http://127.0.0.1:2024").rstrip("/")
    return f"{base}/topology/embed"


def get_topology_files_url() -> str:
    """Backup URL via static file mount."""
    base = os.getenv("LANGGRAPH_PUBLIC_BASE_URL", "http://127.0.0.1:2024").rstrip("/")
    return f"{base}/topology/files/propagation_graph_latest.html"


def get_topology_png_url() -> str:
    """Public URL for PNG topology (renders in LangGraph Studio chat markdown)."""
    base = os.getenv("LANGGRAPH_PUBLIC_BASE_URL", "http://127.0.0.1:2024").rstrip("/")
    return f"{base}/topology/latest.png"


LATEST_GRAPH_PNG = "propagation_graph_latest.png"
LATEST_EMBED_FILE = "propagation_graph_embed.html"

# Vis-network presentation (larger labels for readability)
VIS_FONT_SIZE = 19
VIS_NODE_MARGIN = 28
VIS_NODE_MIN_W = 220
VIS_NODE_MAX_W = 380

_NODE_STYLE = {
    "abnormal": {"bg": "#FEE2E2", "border": "#DC2626", "text": "#7F1D1D", "label": "异常 KPI"},
    "root": {"bg": "#CCFBF1", "border": "#0D9488", "text": "#134E4A", "label": "根因"},
    "intermediate": {"bg": "#E0E7FF", "border": "#4F46E5", "text": "#312E81", "label": "传播节点"},
    "source": {"bg": "#FEF3C7", "border": "#D97706", "text": "#78350F", "label": "上游指标"},
}


def _truncate_node_name(name: str, max_length: int = 30) -> str:
    if len(name) <= max_length:
        return name
    return name[: max_length - 3] + "..."


def _metric_semantic_en(metric: str) -> str:
    name = metric.lower()
    if "error" in name and "rate" in name:
        return "error rate up"
    if "cpu" in name:
        return "CPU saturation"
    if "duration" in name or "latency" in name:
        return "latency spike"
    if "count" in name or "qps" in name:
        return "traffic surge"
    if "memory" in name or "_mem" in name:
        return "memory pressure"
    return "anomaly"


def _node_display_label(metric: str) -> str:
    from app.utils.topology_chat import metric_semantic_label

    return f"{_truncate_node_name(metric, 34)}\n({metric_semantic_label(metric)})"


def _node_display_label_png(metric: str) -> str:
    return f"{_truncate_node_name(metric, 22)}\n({_metric_semantic_en(metric)})"


def _classify_node(
    node: str,
    root_causes: List[str],
    abnormal_kpi: Optional[str],
    in_degree: Dict[str, int],
) -> str:
    root_set = set(root_causes or [])
    if node == abnormal_kpi:
        return "abnormal"
    if node in root_set:
        return "root"
    if in_degree.get(node, 0) == 0:
        return "source"
    return "intermediate"


def _compute_node_levels(nodes: List[str], edges: List[List[str]]) -> Dict[str, int]:
    levels: Dict[str, int] = {}

    def level(node: str, visiting: set) -> int:
        if node in levels:
            return levels[node]
        if node in visiting:
            return 0
        visiting.add(node)
        preds = [e[0] for e in edges if len(e) >= 2 and e[1] == node]
        lv = 0 if not preds else max(level(p, visiting) for p in preds if p in nodes) + 1
        visiting.discard(node)
        levels[node] = lv
        return lv

    for n in nodes:
        level(n, set())
    return levels


def _build_vis_nodes_edges_data(
    nodes: List[str],
    edges: List[List[str]],
    root_causes: List[str],
    abnormal_kpi: Optional[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    in_degree = {n: 0 for n in nodes}
    for edge in edges:
        if len(edge) >= 2:
            in_degree[edge[1]] = in_degree.get(edge[1], 0) + 1

    nodes_data = []
    for node in nodes:
        kind = _classify_node(node, root_causes, abnormal_kpi, in_degree)
        style = _NODE_STYLE[kind]
        nodes_data.append({
            "id": node,
            "label": _node_display_label(node),
            "title": f"{node}\n{style['label']}",
            "shape": "box",
            "margin": VIS_NODE_MARGIN,
            "widthConstraint": {"minimum": VIS_NODE_MIN_W, "maximum": VIS_NODE_MAX_W},
            "color": {
                "background": style["bg"],
                "border": style["border"],
                "highlight": {"background": style["bg"], "border": style["border"]},
            },
            "font": {
                "face": "Microsoft YaHei, PingFang SC, Arial",
                "size": VIS_FONT_SIZE,
                "color": style["text"],
                "align": "center",
            },
            "borderWidth": 2.5,
            "shapeProperties": {"borderRadius": 12},
        })

    edges_data = []
    seen_edges = set()
    for edge in edges:
        if len(edge) >= 2:
            source, target = edge[0], edge[1]
            edge_key = f"{source}->{target}"
            if edge_key not in seen_edges:
                edges_data.append({
                    "from": source,
                    "to": target,
                    "arrows": {"to": {"enabled": True, "scaleFactor": 0.9}},
                    "color": {"color": "#64748B", "highlight": "#334155"},
                    "width": 2.5,
                    "smooth": {
                        "type": "cubicBezier",
                        "forceDirection": "horizontal",
                        "roundness": 0.35,
                    },
                })
                seen_edges.add(edge_key)
    return nodes_data, edges_data


def _vis_layout_metrics(n_nodes: int) -> Tuple[int, int, int, int]:
    canvas_h = min(960, max(680, n_nodes * 150))
    level_sep = max(440, 360 + n_nodes * 28)
    node_spacing = max(300, 240 + n_nodes * 22)
    tree_spacing = max(360, 300 + n_nodes * 24)
    return canvas_h, level_sep, node_spacing, tree_spacing


def _vis_network_init_script(
    container_id: str,
    nodes_data: List[Dict[str, Any]],
    edges_data: List[Dict[str, Any]],
    level_sep: int,
    node_spacing: int,
    tree_spacing: int,
) -> str:
    """vis-network init: large labels, drag nodes after layout, fix position on drag end."""
    return f"""
        var nodes = new vis.DataSet({json.dumps(nodes_data, ensure_ascii=False)});
        var edges = new vis.DataSet({json.dumps(edges_data, ensure_ascii=False)});
        var container = document.getElementById('{container_id}');
        var layoutOptions = {{
            hierarchical: {{
                enabled: true,
                direction: 'LR',
                sortMethod: 'directed',
                levelSeparation: {level_sep},
                nodeSpacing: {node_spacing},
                treeSpacing: {tree_spacing},
                blockShifting: true,
                edgeMinimization: true
            }}
        }};
        var options = {{
            nodes: {{
                shadow: {{ enabled: true, size: 12, x: 2, y: 3, color: 'rgba(15,23,42,0.15)' }}
            }},
            edges: {{ shadow: false }},
            layout: layoutOptions,
            physics: {{
                enabled: true,
                hierarchicalRepulsion: {{
                    centralGravity: 0.0,
                    springLength: {max(320, level_sep - 80)},
                    springConstant: 0.005,
                    nodeDistance: {node_spacing + 60},
                    damping: 0.12,
                    avoidOverlap: 1
                }},
                solver: 'hierarchicalRepulsion',
                stabilization: {{ iterations: 250, updateInterval: 25, fit: true }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100,
                zoomView: true,
                dragView: true,
                dragNodes: true,
                multiselect: false,
                navigationButtons: true,
                keyboard: {{ enabled: true }}
            }}
        }};
        var network = new vis.Network(container, {{ nodes: nodes, edges: edges }}, options);

        function fitView() {{
            network.fit({{ animation: {{ duration: 500, easingFunction: 'easeInOutQuad' }} }});
        }}

        function enableFreeDrag() {{
            network.setOptions({{
                physics: false,
                layout: {{ hierarchical: {{ enabled: false }} }}
            }});
        }}

        function relayout() {{
            nodes.getIds().forEach(function(id) {{
                nodes.update({{ id: id, fixed: false }});
            }});
            network.setOptions({{
                physics: {{ enabled: true }},
                layout: layoutOptions
            }});
            network.once('stabilizationIterationsDone', function() {{
                enableFreeDrag();
                fitView();
            }});
            network.stabilize(250);
        }}

        network.once('stabilizationIterationsDone', function() {{
            enableFreeDrag();
            fitView();
        }});

        network.on('dragEnd', function(params) {{
            if (params.nodes && params.nodes.length > 0) {{
                var nid = params.nodes[0];
                var pos = network.getPositions([nid]);
                nodes.update({{
                    id: nid,
                    x: pos[nid].x,
                    y: pos[nid].y,
                    fixed: {{ x: true, y: true }}
                }});
            }}
        }});

        var btnFit = document.getElementById('btnFit');
        var btnRelayout = document.getElementById('btnRelayout');
        if (btnFit) btnFit.addEventListener('click', fitView);
        if (btnRelayout) btnRelayout.addEventListener('click', relayout);
    """


def visualize_propagation_graph(
    edges: List[List[str]],
    root_causes: List[str],
    abnormal_kpi: Optional[str] = None,
    output_format: str = "html",
    output_dir: str = "outputs/graphs"
) -> Dict[str, Any]:
    """
    Visualize the fault propagation graph with different colors for KPI and root cause nodes.

    Args:
        edges: List of directed edges [[source, target], ...]
        root_causes: List of root cause metrics
        abnormal_kpi: The abnormal KPI metric (highlighted)
        output_format: Output format - "html" (interactive) or "png" (static image)
        output_dir: Directory to save output files

    Returns:
        Dictionary with visualization result
    """
    try:
        chain_edges = filter_propagation_chain_edges(edges, root_causes, abnormal_kpi)
        directed_edges = orient_propagation_edges(chain_edges, root_causes, abnormal_kpi)
        propagation_paths = list_propagation_paths(edges, root_causes, abnormal_kpi)
        if len(directed_edges) < len(chain_edges):
            logger.info(
                f"Oriented propagation edges: {len(directed_edges)} directed "
                f"(from {len(chain_edges)} chain / {len(edges)} raw)"
            )
        logger.info(
            f"Generating propagation topology: {len(directed_edges)} directed edges, "
            f"{len(root_causes)} root causes"
        )

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Extract all unique nodes
        nodes = set()
        for edge in directed_edges:
            if len(edge) >= 2:
                nodes.add(edge[0])
                nodes.add(edge[1])

        nodes = sorted(list(nodes))
        logger.info(f"Found {len(nodes)} nodes in the graph")

        # Generate visualization based on format
        if output_format == "html":
            result = _generate_html_visualization(
                nodes, directed_edges, root_causes, abnormal_kpi, output_path, propagation_paths
            )
        elif output_format == "png":
            result = _generate_png_visualization(
                nodes, directed_edges, root_causes, abnormal_kpi, output_path
            )
        else:
            result = _generate_html_visualization(
                nodes, directed_edges, root_causes, abnormal_kpi, output_path, propagation_paths
            )

        view_url = get_topology_view_url()
        payload: Dict[str, Any] = {
            "success": True,
            "format": output_format,
            "nodes": nodes,
            "edges": directed_edges,
            "chain_edges": chain_edges,
            "all_edges": edges,
            "propagation_paths": propagation_paths,
            "root_causes": root_causes,
            "abnormal_kpi": abnormal_kpi,
            "view_url": view_url,
            **result,
        }
        payload["embed_url"] = get_topology_embed_url()
        return payload

    except Exception as e:
        logger.error(f"Failed to generate visualization: {e}")
        return {
            "success": False,
            "error": str(e),
            "format": output_format
        }


def _generate_html_visualization(
    nodes: List[str],
    edges: List[List[str]],
    root_causes: List[str],
    abnormal_kpi: Optional[str],
    output_path: Path,
    propagation_paths: Optional[List[List[str]]] = None,
) -> Dict[str, Any]:
    """Generate interactive HTML visualization using vis.js network library."""

    # Create unique filename
    import time
    timestamp = int(time.time() * 1000)
    filename = f"propagation_graph_{timestamp}.html"
    filepath = output_path / filename

    nodes_data, edges_data = _build_vis_nodes_edges_data(
        nodes, edges, root_causes, abnormal_kpi
    )

    paths_html = ""
    if propagation_paths:
        from app.utils.topology_chat import format_propagation_path_line

        paths_html = "<h3>根因传播链</h3><ol>" + "".join(
            f"<li>{format_propagation_path_line(i, path)}</li>"
            for i, path in enumerate(propagation_paths[:15], 1)
        ) + "</ol>"
        if len(propagation_paths) > 15:
            paths_html += f"<p>… 另有 {len(propagation_paths) - 15} 条路径</p>"

    view_url = get_topology_view_url()
    embed_url = get_topology_embed_url()
    n_nodes = max(len(nodes), 1)
    canvas_h, level_sep, node_spacing, tree_spacing = _vis_layout_metrics(n_nodes)
    vis_script = _vis_network_init_script(
        "mynetwork", nodes_data, edges_data, level_sep, node_spacing, tree_spacing
    )

    # Generate HTML with vis.js
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>根因传播拓扑图</title>
    <script type="text/javascript" src="/topology/static/vis-network.min.js"></script>
    <style>
        body {{
            font-family: 'Microsoft YaHei', 'PingFang SC', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(160deg, #f8fafc 0%, #eef2ff 100%);
        }}
        .toolbar {{
            margin: 12px 0 8px;
            padding: 12px 16px;
            background: #fff;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            color: #334155;
            font-size: 16px;
            line-height: 1.5;
        }}
        .toolbar a {{
            color: #2563eb;
            font-weight: 600;
        }}
        .toolbar kbd {{
            background: #f1f5f9;
            border: 1px solid #cbd5e1;
            border-radius: 4px;
            padding: 2px 6px;
            font-size: 12px;
        }}
        #mynetwork {{
            width: 100%;
            height: {canvas_h}px;
            min-height: 620px;
            border: 1px solid #e2e8f0;
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 24px rgba(15, 23, 42, 0.08);
        }}
        .legend {{
            margin-top: 20px;
            padding: 15px;
            background-color: white;
            border-radius: 8px;
            border: 1px solid #ddd;
        }}
        .legend-item {{
            display: inline-block;
            margin-right: 20px;
            margin-bottom: 10px;
        }}
        .legend-color {{
            display: inline-block;
            width: 20px;
            height: 20px;
            margin-right: 5px;
            border-radius: 3px;
            vertical-align: middle;
        }}
        .stats {{
            margin-top: 20px;
            padding: 15px;
            background-color: white;
            border-radius: 8px;
            border: 1px solid #ddd;
        }}
        h1 {{
            color: #333;
            margin-bottom: 10px;
        }}
        .info {{
            color: #666;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <h1>根因传播拓扑图</h1>
    <div class="info">
        <p><strong>异常指标:</strong> {abnormal_kpi or '未指定'}</p>
        <p><strong>根因数量:</strong> {len(root_causes)}</p>
        <p><strong>节点数:</strong> {len(nodes)} | <strong>传播边数:</strong> {len(edges)}</p>
        <p><strong>访问地址:</strong> <a href="{view_url}">{view_url}</a></p>
    </div>

    {paths_html}

    <div class="toolbar">
        <strong>交互操作：</strong> 鼠标<strong>拖动节点</strong>可自由调整位置 · 拖动画布平移 · 滚轮缩放 ·
        <button type="button" id="btnFit" style="margin-left:8px;padding:6px 12px;cursor:pointer;font-size:14px;">适应窗口</button>
        <button type="button" id="btnRelayout" style="margin-left:6px;padding:6px 12px;cursor:pointer;font-size:14px;">重新布局</button>
        <br/>
        <a href="{embed_url}" target="_blank">全屏可拖拽拓扑（推荐）</a>
    </div>

    <div id="mynetwork"></div>

    <div class="legend">
        <h3>图例</h3>
        <div class="legend-item">
            <span class="legend-color" style="background:#FEE2E2;border:2px solid #DC2626;"></span>
            <span>异常 KPI</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background:#CCFBF1;border:2px solid #0D9488;"></span>
            <span>根因指标</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background:#E0E7FF;border:2px solid #4F46E5;"></span>
            <span>传播节点</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background:#FEF3C7;border:2px solid #D97706;"></span>
            <span>上游指标</span>
        </div>
    </div>

    <div class="stats">
        <h3>根因指标</h3>
        <ul>
            {"".join(f"<li><strong>{rc}</strong></li>" for rc in root_causes[:10])}
            {f"<li>... and {len(root_causes) - 10} more</li>" if len(root_causes) > 10 else ""}
        </ul>
    </div>

    <script type="text/javascript">{vis_script}</script>
</body>
</html>"""

    # Write HTML file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)

    latest_path = output_path / LATEST_GRAPH_FILENAME
    shutil.copy2(filepath, latest_path)

    embed_html = _build_embed_html(
        nodes_data, edges_data, n_nodes, level_sep, node_spacing, tree_spacing
    )
    embed_path = output_path / LATEST_EMBED_FILE
    with open(embed_path, "w", encoding="utf-8") as ef:
        ef.write(embed_html)
    logger.info(f"Generated HTML visualization: {filepath} (latest: {latest_path}, embed: {embed_path})")

    # Also return as base64 for inline display
    base64_html = base64.b64encode(html_content.encode('utf-8')).decode('ascii')

    return {
        "filepath": str(filepath),
        "filename": filename,
        "html_content": html_content,
        "base64_content": base64_html,
        "embed_url": get_topology_embed_url(),
    }


def _build_embed_html(
    nodes_data: List[Dict[str, Any]],
    edges_data: List[Dict[str, Any]],
    n_nodes: int,
    level_sep: int,
    node_spacing: int,
    tree_spacing: int,
) -> str:
    """Fullscreen page optimized for dragging nodes with large labels."""
    canvas_h = min(900, max(720, n_nodes * 160))
    vis_script = _vis_network_init_script(
        "mynetwork", nodes_data, edges_data, level_sep, node_spacing, tree_spacing
    )
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>根因传播拓扑 · 可拖拽</title>
  <script src="/topology/static/vis-network.min.js"></script>
  <style>
    html, body {{ margin: 0; height: 100%; background: #f1f5f9; font-family: "Microsoft YaHei", Arial, sans-serif; }}
    #toolbar {{
      padding: 14px 18px; background: #fff; border-bottom: 2px solid #e2e8f0;
      font-size: 17px; color: #1e293b; line-height: 1.6;
    }}
    #toolbar strong {{ color: #0f172a; }}
  </style>
</head>
<body>
  <div id="toolbar">
    <strong>可拖拽传播拓扑</strong> — 直接<strong>用鼠标拖动任意节点</strong>调整位置；拖动画布平移；滚轮缩放。
    <button type="button" id="btnFit" style="margin-left:12px;padding:8px 14px;font-size:15px;cursor:pointer;">适应窗口</button>
    <button type="button" id="btnRelayout" style="margin-left:8px;padding:8px 14px;font-size:15px;cursor:pointer;">重新布局</button>
  </div>
  <div id="mynetwork" style="width:100%;height:calc(100vh - 72px);min-height:{canvas_h}px;background:#fff;"></div>
  <script type="text/javascript">{vis_script}</script>
</body>
</html>"""


def _draw_png_metric_node(
    ax,
    xy: Tuple[float, float],
    node: str,
    kind: str,
    box_w: float = 0.52,
    box_h: float = 0.20,
) -> Tuple[float, float, float, float]:
    """Draw rounded metric card; returns (x0, y0, x1, y1) bbox."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    style = _NODE_STYLE[kind]
    x, y = xy
    x0, y0 = x - box_w / 2, y - box_h / 2
    patch = FancyBboxPatch(
        (x0, y0),
        box_w,
        box_h,
        boxstyle="round,pad=0.012,rounding_size=0.04",
        facecolor=style["bg"],
        edgecolor=style["border"],
        linewidth=2.2,
        zorder=3,
    )
    ax.add_patch(patch)
    lines = _node_display_label_png(node).split("\n")
    ax.text(
        x,
        y + 0.018,
        lines[0],
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=style["text"],
        zorder=4,
    )
    if len(lines) > 1:
        ax.text(
            x,
            y - 0.032,
            lines[1],
            ha="center",
            va="center",
            fontsize=11,
            color=style["text"],
            alpha=0.9,
            zorder=4,
        )
    return x0, y0, x0 + box_w, y0 + box_h


def _draw_png_edge(
    ax,
    src_box: Tuple[float, float, float, float],
    dst_box: Tuple[float, float, float, float],
) -> None:
    from matplotlib.patches import FancyArrowPatch

    sx0, sy0, sx1, sy1 = src_box
    dx0, dy0, dx1, dy1 = dst_box
    start = (sx1, (sy0 + sy1) / 2)
    end = (dx0, (dy0 + dy1) / 2)
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=2,
        color="#64748B",
        connectionstyle="arc3,rad=0.08",
        zorder=2,
    )
    ax.add_patch(arrow)


def _generate_png_visualization(
    nodes: List[str],
    edges: List[List[str]],
    root_causes: List[str],
    abnormal_kpi: Optional[str],
    output_path: Path
) -> Dict[str, Any]:
    """Generate static PNG: one rounded card per metric, left-to-right propagation."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import networkx as nx

        logger.info("Generating PNG visualization (metric cards)")

        in_degree = {n: 0 for n in nodes}
        for edge in edges:
            if len(edge) >= 2:
                in_degree[edge[1]] = in_degree.get(edge[1], 0) + 1

        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        for edge in edges:
            if len(edge) >= 2:
                G.add_edge(edge[0], edge[1])

        levels = _compute_node_levels(nodes, edges)
        for n in nodes:
            G.nodes[n]["subset"] = levels.get(n, 0)

        pos = nx.multipartite_layout(
            G, subset_key="subset", align="vertical", scale=4.2, center=None
        )

        n_levels = len(set(levels.values())) or 1
        fig_w = max(16, n_levels * 5.5)
        fig_h = max(9, len(nodes) * 0.85)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor("#f8fafc")
        ax.set_facecolor("#f8fafc")
        ax.set_title(
            "Root Cause Propagation Topology",
            fontsize=18,
            fontweight="bold",
            color="#0f172a",
            pad=18,
        )

        node_boxes: Dict[str, Tuple[float, float, float, float]] = {}
        for node in nodes:
            if node not in pos:
                continue
            kind = _classify_node(node, root_causes, abnormal_kpi, in_degree)
            node_boxes[node] = _draw_png_metric_node(ax, pos[node], node, kind)

        for edge in edges:
            if len(edge) >= 2 and edge[0] in node_boxes and edge[1] in node_boxes:
                _draw_png_edge(ax, node_boxes[edge[0]], node_boxes[edge[1]])

        _legend_en = {
            "abnormal": "Abnormal KPI",
            "root": "Root cause",
            "intermediate": "Propagation",
            "source": "Upstream",
        }
        legend_elements = [
            mpatches.Patch(
                facecolor=_NODE_STYLE[k]["bg"],
                edgecolor=_NODE_STYLE[k]["border"],
                linewidth=2,
                label=_legend_en[k],
            )
            for k in ("root", "intermediate", "abnormal", "source")
        ]
        ax.legend(handles=legend_elements, loc="upper left", frameon=True, fancybox=True, fontsize=9)

        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        margin_x, margin_y = 0.85, 0.5
        ax.set_xlim(min(xs) - margin_x, max(xs) + margin_x)
        ax.set_ylim(min(ys) - margin_y, max(ys) + margin_y)
        ax.axis("off")
        ax.set_aspect("equal", adjustable="box")

        timestamp = int(__import__('time').time() * 1000)
        filename = f"propagation_graph_{timestamp}.png"
        filepath = output_path / filename

        plt.savefig(
            filepath,
            dpi=180,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
        plt.close(fig)

        latest_png = output_path / LATEST_GRAPH_PNG
        shutil.copy2(filepath, latest_png)
        logger.info(f"Generated PNG visualization: {filepath} (latest: {latest_png})")

        # Convert to base64 for inline display
        with open(filepath, 'rb') as f:
            base64_png = base64.b64encode(f.read()).decode('ascii')

        return {
            "filepath": str(filepath),
            "filename": filename,
            "base64_content": base64_png,
            "png_url": get_topology_png_url(),
        }

    except ImportError as e:
        logger.warning(f"matplotlib not available: {e}")
        # Fall back to HTML
        return _generate_html_visualization(nodes, edges, root_causes, abnormal_kpi, output_path)


def _calculate_hierarchical_positions(nodes: List[str], edges: List[List[str]]) -> Dict[str, Tuple[float, float]]:
    """Calculate hierarchical positions for nodes."""
    # Build adjacency
    out_edges = {node: [] for node in nodes}
    in_degree = {node: 0 for node in nodes}

    for edge in edges:
        if len(edge) >= 2:
            source, target = edge[0], edge[1]
            if source in out_edges:
                out_edges[source].append(target)
            if target in in_degree:
                in_degree[target] = in_degree.get(target, 0) + 1

    # Calculate levels (longest path from source)
    levels = {}
    visited = set()

    def get_level(node):
        if node in levels:
            return levels[node]
        if node in visited:
            return 0  # Cycle detected
        visited.add(node)

        max_child_level = -1
        for child in out_edges.get(node, []):
            if child in nodes:
                child_level = get_level(child)
                max_child_level = max(max_child_level, child_level)

        levels[node] = max_child_level + 1
        return levels[node]

    # Calculate levels for all nodes
    for node in nodes:
        if node not in levels:
            visited = set()
            get_level(node)

    # Group nodes by level
    level_groups = {}
    for node, level in levels.items():
        if level not in level_groups:
            level_groups[level] = []
        level_groups[level].append(node)

    # Assign positions
    positions = {}
    for level in sorted(level_groups.keys()):
        nodes_in_level = sorted(level_groups[level])
        level_width = len(nodes_in_level)
        for i, node in enumerate(nodes_in_level):
            x = level
            y = (i - level_width / 2.0) / max(level_width, 1) * 2  # Spread vertically
            positions[node] = (x, y)

    return positions


def format_graph_for_llm(edges: List[List[str]], root_causes: List[str], abnormal_kpi: Optional[str]) -> str:
    """Format graph data for LLM consumption in prompt.

    Args:
        edges: List of directed edges
        root_causes: List of root cause nodes
        abnormal_kpi: The abnormal KPI node

    Returns:
        Formatted string description of the graph
    """
    lines = ["## 故障传播图分析结果\n"]

    if abnormal_kpi:
        lines.append(f"**异常指标**: {abnormal_kpi}\n")

    lines.append(f"**发现根因**: {len(root_causes)} 个\n")
    if root_causes:
        for i, rc in enumerate(root_causes[:10], 1):
            lines.append(f"{i}. {rc}")
        if len(root_causes) > 10:
            lines.append(f"... 及其他 {len(root_causes) - 10} 个\n")

    lines.append(f"\n**因果关系**: {len(edges)} 条传播路径\n")

    # Group edges by source
    edges_by_source = {}
    for edge in edges:
        if len(edge) >= 2:
            source = edge[0]
            target = edge[1]
            if source not in edges_by_source:
                edges_by_source[source] = []
            edges_by_source[source].append(target)

    # Display propagation paths
    lines.append("\n**关键传播路径**:\n")
    for source, targets in sorted(edges_by_source.items())[:15]:
        targets_str = ", ".join(targets[:5])
        if len(targets) > 5:
            targets_str += f" ... (+{len(targets)-5} more)"
        lines.append(f"• {source} → [{targets_str}]")

    lines.append(f"\n**传播层级分析**:\n")

    # Calculate levels
    out_edges = {node: [] for node in set([e[0] for e in edges] + [e[1] for e in edges if len(e) >= 2])}
    for edge in edges:
        if len(edge) >= 2:
            out_edges[edge[0]].append(edge[1])

    levels = {}
    def get_level(node, visited=None):
        if visited is None:
            visited = set()
        if node in levels:
            return levels[node]
        if node in visited:
            return 0
        visited.add(node)

        max_level = 0
        for child in out_edges.get(node, []):
            level = get_level(child, visited)
            max_level = max(max_level, level + 1)

        levels[node] = max_level
        return max_level

    for node in out_edges:
        if node not in levels:
            get_level(node)

    # Group by level
    level_groups = {}
    for node, level in levels.items():
        if level not in level_groups:
            level_groups[level] = []
        level_groups[level].append(node)

    for level in sorted(level_groups.keys()):
        nodes_at_level = level_groups[level]
        lines.append(f"- 层级 {level}: {', '.join(nodes_at_level[:10])}" + (f" ... (+{len(nodes_at_level)-10} more)" if len(nodes_at_level) > 10 else ""))

    return "\n".join(lines)
