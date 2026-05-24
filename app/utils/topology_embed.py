"""Build chat-embeddable topology (iframe) for LangGraph Studio."""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, List, Optional, Tuple

from app.tools.graph_visualization_tool import (
    _classify_node,
    _compute_node_levels,
    _node_display_label,
    _NODE_STYLE,
)


def build_vis_datasets(
    nodes: List[str],
    edges: List[List[str]],
    root_causes: List[str],
    abnormal_kpi: Optional[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """Vis-network node/edge datasets (same styling as full HTML page)."""
    in_degree = {n: 0 for n in nodes}
    for edge in edges:
        if len(edge) >= 2:
            in_degree[edge[1]] = in_degree.get(edge[1], 0) + 1

    nodes_data: List[Dict[str, Any]] = []
    for node in nodes:
        kind = _classify_node(node, root_causes, abnormal_kpi, in_degree)
        style = _NODE_STYLE[kind]
        nodes_data.append(
            {
                "id": node,
                "label": _node_display_label(node),
                "title": f"{node}\n{style['label']}",
                "shape": "box",
                "margin": 18,
                "widthConstraint": {"minimum": 150, "maximum": 240},
                "color": {
                    "background": style["bg"],
                    "border": style["border"],
                    "highlight": {"background": style["bg"], "border": style["border"]},
                },
                "font": {
                    "face": "Microsoft YaHei, PingFang SC, Arial",
                    "size": 13,
                    "color": style["text"],
                    "align": "center",
                },
                "borderWidth": 2,
                "shapeProperties": {"borderRadius": 10},
            }
        )

    edges_data: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for edge in edges:
        if len(edge) >= 2:
            source, target = edge[0], edge[1]
            key = f"{source}->{target}"
            if key not in seen:
                edges_data.append(
                    {
                        "from": source,
                        "to": target,
                        "arrows": {"to": {"enabled": True, "scaleFactor": 0.85}},
                        "color": {"color": "#64748B", "highlight": "#334155"},
                        "width": 2.5,
                        "smooth": {
                            "type": "cubicBezier",
                            "forceDirection": "horizontal",
                            "roundness": 0.35,
                        },
                    }
                )
                seen.add(key)

    n_nodes = max(len(nodes), 1)
    return nodes_data, edges_data, n_nodes


def build_interactive_topology_html(
    nodes_data: List[Dict[str, Any]],
    edges_data: List[Dict[str, Any]],
    n_nodes: int,
) -> str:
    """Minimal HTML document for iframe embed (vis.js via CDN, draggable nodes)."""
    level_sep = max(400, 320 + n_nodes * 25)
    node_spacing = max(260, 200 + n_nodes * 18)
    tree_spacing = max(320, 260 + n_nodes * 20)
    nodes_json = json.dumps(nodes_data, ensure_ascii=False)
    edges_json = json.dumps(edges_data, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
  html, body {{ margin: 0; height: 100%; background: #f8fafc; font-family: "Microsoft YaHei", Arial, sans-serif; }}
  #toolbar {{ padding: 8px 12px; font-size: 13px; color: #475569; background: #fff; border-bottom: 1px solid #e2e8f0; }}
  #g {{ width: 100%; height: calc(100% - 40px); }}
  button {{ margin-left: 6px; padding: 4px 10px; cursor: pointer; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
</head><body>
<div id="toolbar"><strong>传播拓扑</strong> · 拖动节点 · 滚轮缩放
  <button type="button" id="fit">适应</button>
  <button type="button" id="relayout">重排</button>
</div>
<div id="g"></div>
<script>
(function() {{
  var nodes = new vis.DataSet({nodes_json});
  var edges = new vis.DataSet({edges_json});
  var layoutOptions = {{
    hierarchical: {{
      enabled: true, direction: 'LR', sortMethod: 'directed',
      levelSeparation: {level_sep}, nodeSpacing: {node_spacing}, treeSpacing: {tree_spacing},
      blockShifting: true, edgeMinimization: true
    }}
  }};
  var options = {{
    nodes: {{ shadow: {{ enabled: true, size: 8, x: 2, y: 2, color: 'rgba(15,23,42,0.12)' }} }},
    layout: layoutOptions,
    physics: {{
      enabled: true,
      hierarchicalRepulsion: {{
        centralGravity: 0, springLength: {max(280, level_sep - 80)},
        springConstant: 0.006, nodeDistance: {node_spacing + 40}, damping: 0.12, avoidOverlap: 1
      }},
      solver: 'hierarchicalRepulsion',
      stabilization: {{ iterations: 200, fit: true }}
    }},
    interaction: {{ dragNodes: true, dragView: true, zoomView: true, navigationButtons: true }}
  }};
  var net = new vis.Network(document.getElementById('g'), {{ nodes: nodes, edges: edges }}, options);
  function fit() {{ net.fit({{ animation: {{ duration: 500 }} }}); }}
  net.once('stabilizationIterationsDone', function() {{ net.setOptions({{ physics: false }}); fit(); }});
  document.getElementById('fit').onclick = fit;
  document.getElementById('relayout').onclick = function() {{
    net.setOptions({{ physics: {{ enabled: true }}, layout: layoutOptions }});
    net.once('stabilizationIterationsDone', function() {{ net.setOptions({{ physics: false }}); fit(); }});
    net.stabilize(200);
  }};
}})();
</script>
</body></html>"""


def build_chat_iframe_markdown(html_doc: str, height: int = 620) -> str:
    """Embed interactive HTML in chat via data-URI iframe (no external page)."""
    # quote via base64 to avoid srcdoc escaping issues
    b64 = base64.b64encode(html_doc.encode("utf-8")).decode("ascii")
    return (
        f'\n<iframe title="根因传播拓扑图" '
        f'src="data:text/html;base64,{b64}" '
        f'width="100%" height="{height}" '
        f'style="border:1px solid #e2e8f0;border-radius:12px;background:#fff;min-height:{height}px;" '
        f'sandbox="allow-scripts allow-same-origin"></iframe>\n'
    )


def build_chat_topology_embed(graph_viz: Dict[str, Any]) -> str:
    """Full chat appendix: paths + interactive iframe embed."""
    if not graph_viz or not graph_viz.get("success"):
        return ""

    nodes = graph_viz.get("nodes") or []
    edges = graph_viz.get("edges") or []
    root_causes = graph_viz.get("root_causes") or []
    abnormal_kpi = graph_viz.get("abnormal_kpi")

    from app.utils.topology_chat import format_propagation_paths_section

    parts: List[str] = []
    paths = graph_viz.get("propagation_paths") or []
    path_section = format_propagation_paths_section(paths)
    if path_section:
        parts.append(path_section)

    if nodes and edges:
        nodes_data, edges_data, n_nodes = build_vis_datasets(
            nodes, edges, root_causes, abnormal_kpi
        )
        html_doc = build_interactive_topology_html(nodes_data, edges_data, n_nodes)
        canvas_h = min(720, max(560, n_nodes * 120))
        parts.append("\n### 传播拓扑图（可拖拽 · 已嵌入对话）\n")
        parts.append(build_chat_iframe_markdown(html_doc, height=canvas_h))

    return "".join(parts)
