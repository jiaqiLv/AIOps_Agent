"""Utilities for root-cause propagation chain extraction."""

from typing import Dict, List, Optional, Set, Tuple

EdgePair = Tuple[str, str]


def filter_propagation_chain_edges(
    edges: List[List[str]],
    root_causes: List[str],
    abnormal_kpi: Optional[str] = None,
) -> List[List[str]]:
    """Keep edges on paths leading to the abnormal KPI (upstream propagation chain)."""
    if not edges:
        return []

    normalized: List[Tuple[str, str]] = []
    for edge in edges:
        if len(edge) >= 2 and edge[0] and edge[1]:
            normalized.append((str(edge[0]), str(edge[1])))

    if not normalized:
        return []

    if not abnormal_kpi:
        return [[s, t] for s, t in normalized]

    preds: Dict[str, List[str]] = {}
    for source, target in normalized:
        preds.setdefault(target, []).append(source)

    chain_nodes: Set[str] = {abnormal_kpi}
    queue = [abnormal_kpi]
    while queue:
        node = queue.pop(0)
        for parent in preds.get(node, []):
            if parent not in chain_nodes:
                chain_nodes.add(parent)
                queue.append(parent)

    for root in root_causes or []:
        chain_nodes.add(root)

    filtered = [
        [source, target]
        for source, target in normalized
        if source in chain_nodes and target in chain_nodes
    ]
    return filtered if filtered else [[s, t] for s, t in normalized]


def _upstream_rank_from_abnormal(
    edges: List[EdgePair],
    abnormal_kpi: str,
) -> Dict[str, int]:
    """Longest-hop distance from abnormal KPI backward (higher = more upstream).

    Uses Bellman-Ford style relaxation so cycles in PC graphs do not loop forever.
    """
    nodes: Set[str] = {abnormal_kpi}
    for source, target in edges:
        nodes.add(source)
        nodes.add(target)

    rank: Dict[str, int] = {abnormal_kpi: 0}
    n = max(len(nodes), 1)
    for _ in range(n):
        updated = False
        for source, target in edges:
            if target not in rank or source == abnormal_kpi:
                continue
            candidate = rank[target] + 1
            if source not in rank or candidate > rank[source]:
                rank[source] = candidate
                updated = True
        if not updated:
            break
    rank[abnormal_kpi] = 0
    return rank


def _pick_directed_edge(
    u: str,
    v: str,
    rank: Dict[str, int],
    root_causes: List[str],
) -> EdgePair:
    """Choose one direction for a conflicting pair (upstream -> downstream)."""
    ru, rv = rank.get(u), rank.get(v)
    root_set = set(root_causes or [])

    if ru is not None and rv is not None and ru != rv:
        return (u, v) if ru > rv else (v, u)

    if u in root_set and v not in root_set:
        return (u, v)
    if v in root_set and u not in root_set:
        return (v, u)

    if ru is not None and rv is None:
        return (u, v)
    if rv is not None and ru is None:
        return (v, u)

    return (u, v) if u <= v else (v, u)


def orient_propagation_edges(
    edges: List[List[str]],
    root_causes: List[str],
    abnormal_kpi: Optional[str] = None,
) -> List[List[str]]:
    """
    Collapse bidirectional pairs to a single propagation direction (upstream -> abnormal).

    PC / correlation graphs may contain both A->B and B->A; visualization and reports
    should show a directed acyclic propagation view with one arrow per node pair.
    """
    normalized: List[EdgePair] = []
    seen: Set[EdgePair] = set()
    for edge in edges:
        if len(edge) >= 2 and edge[0] and edge[1]:
            pair = (str(edge[0]), str(edge[1]))
            if pair not in seen:
                seen.add(pair)
                normalized.append(pair)

    if not normalized:
        return []

    if not abnormal_kpi:
        undirected: Dict[Tuple[str, str], EdgePair] = {}
        for u, v in normalized:
            key = tuple(sorted((u, v)))
            if key not in undirected:
                undirected[key] = (u, v)
        return [[s, t] for s, t in undirected.values()]

    rank = _upstream_rank_from_abnormal(normalized, abnormal_kpi)
    edge_set = set(normalized)
    oriented: List[List[str]] = []
    handled: Set[Tuple[str, str]] = set()

    for u, v in normalized:
        key = tuple(sorted((u, v)))
        if key in handled:
            continue
        handled.add(key)
        if (v, u) in edge_set:
            source, target = _pick_directed_edge(u, v, rank, root_causes)
        else:
            source, target = u, v
        oriented.append([source, target])

    return oriented


def list_propagation_paths(
    edges: List[List[str]],
    root_causes: List[str],
    abnormal_kpi: Optional[str] = None,
    max_paths: int = 20,
) -> List[List[str]]:
    """Enumerate simple paths from root causes toward abnormal_kpi for reporting."""
    chain_edges = orient_propagation_edges(
        filter_propagation_chain_edges(edges, root_causes, abnormal_kpi),
        root_causes,
        abnormal_kpi,
    )
    if not chain_edges or not abnormal_kpi:
        return []

    adj: Dict[str, List[str]] = {}
    for source, target in (
        (e[0], e[1]) for e in chain_edges if len(e) >= 2
    ):
        adj.setdefault(source, []).append(target)

    paths: List[List[str]] = []
    seen: Set[Tuple[str, ...]] = set()

    def dfs(node: str, path: List[str]) -> None:
        if len(paths) >= max_paths:
            return
        if node == abnormal_kpi and len(path) > 1:
            key = tuple(path)
            if key not in seen:
                seen.add(key)
                paths.append(path.copy())
            return
        for nxt in adj.get(node, []):
            if nxt in path:
                continue
            path.append(nxt)
            dfs(nxt, path)
            path.pop()

    for root in root_causes or []:
        if root in adj or root == abnormal_kpi:
            dfs(root, [root])

    return paths
