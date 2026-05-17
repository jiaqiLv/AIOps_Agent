"""
Causal graph subtraction for extracting root cause subgraphs.

This module implements the graph subtraction process from the RADICE paper
to extract the causal subgraph connecting root causes to the performance metric.
"""

import networkx as nx
from typing import List, Set, Dict, Optional


def extract_root_cause_subgraph(
    graph: nx.DiGraph,
    performance_metric: str,
    candidate_root_causes: List[str],
    node_levels: Optional[Dict[str, int]] = None,
    correlation_scores: Optional[Dict[str, float]] = None,
    max_path_length: int = 10
) -> nx.DiGraph:
    """
    Extract root cause causal subgraph using graph subtraction.

    Args:
        graph: Causal graph
        performance_metric: Performance metric node name
        candidate_root_causes: List of candidate root cause names
        node_levels: Optional level assignments
        correlation_scores: Optional correlation scores for tie-breaking
        max_path_length: Maximum path length for subgraph extraction

    Returns:
        Subgraph containing causal paths
    """
    # Sort candidates
    sorted_candidates = _sort_candidates(
        candidate_root_causes,
        performance_metric,
        node_levels,
        correlation_scores
    )

    # Track subgraph components
    subgraph_nodes = {performance_metric}
    subgraph_edges = set()
    valid_candidates = set()

    # Process each candidate
    for candidate in sorted_candidates:
        if candidate == performance_metric:
            continue

        # Check if path exists using NetworkX
        if not nx.has_path(graph, candidate, performance_metric):
            continue

        # Check path length constraint
        try:
            shortest_path_length = nx.shortest_path_length(graph, candidate, performance_metric)
            if shortest_path_length > max_path_length:
                continue
        except nx.NetworkXNoPath:
            continue

        valid_candidates.add(candidate)

        # Find all simple paths
        try:
            all_paths = list(nx.all_simple_paths(
                graph,
                candidate,
                performance_metric,
                cutoff=max_path_length
            ))
        except nx.NetworkXNoPath:
            continue

        if not all_paths:
            continue

        # Select best path
        best_path = _select_best_path(all_paths, valid_candidates)

        # Add path to subgraph
        for i in range(len(best_path) - 1):
            u = best_path[i]
            v = best_path[i + 1]
            subgraph_nodes.add(u)
            subgraph_nodes.add(v)
            subgraph_edges.add((u, v))

    # Add intermediate connections between nodes in subgraph
    for u in list(subgraph_nodes):
        for v in graph.successors(u):
            if v in subgraph_nodes:
                subgraph_edges.add((u, v))

    # Create subgraph
    subgraph = nx.DiGraph()
    subgraph.add_nodes_from(subgraph_nodes)

    for u, v in subgraph_edges:
        if graph.has_edge(u, v):
            subgraph.add_edge(u, v, **graph[u][v])

    return subgraph


def _sort_candidates(
    candidates: List[str],
    performance_metric: str,
    node_levels: Optional[Dict[str, int]],
    correlation_scores: Optional[Dict[str, float]]
) -> List[str]:
    """Sort candidates by distance to performance metric."""
    if node_levels:
        return sorted(
            candidates,
            key=lambda x: (
                node_levels.get(x, 0),
                correlation_scores.get(x, 0) if correlation_scores else 0
            ),
            reverse=True
        )
    elif correlation_scores:
        return sorted(
            candidates,
            key=lambda x: correlation_scores.get(x, 0),
            reverse=True
        )
    return candidates.copy()


def _select_best_path(paths: List[List[str]], valid_candidates: Set[str]) -> List[str]:
    """Select best path from list of paths."""
    if len(paths) == 1:
        return paths[0]

    scored_paths = []
    for path in paths:
        candidate_count = sum(1 for node in path if node in valid_candidates)
        path_length = len(path)
        scored_paths.append((path, candidate_count, -path_length))

    scored_paths.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return scored_paths[0][0]


def classify_subgraph_nodes(
    subgraph: nx.DiGraph,
    performance_metric: str,
    root_causes: Set[str]
) -> Dict[str, str]:
    """
    Classify nodes in the subgraph.

    Args:
        subgraph: Root cause subgraph
        performance_metric: Performance metric name
        root_causes: Set of root cause names

    Returns:
        Dictionary mapping node names to classifications
    """
    classifications = {}

    for node in subgraph.nodes():
        if node == performance_metric:
            classifications[node] = 'performance'
        elif node in root_causes:
            classifications[node] = 'root_cause'
        else:
            classifications[node] = 'intermediate'

    return classifications