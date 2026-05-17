"""
Evaluation metrics for causal discovery and root cause analysis.

This module provides common metrics for evaluating causal graphs and
root cause localization results, following the RQ1 benchmark implementation.
"""

from typing import List, Set, Tuple, Dict
from itertools import combinations
import networkx as nx


def compute_precision_recall_f1(predicted: Set, ground_truth: Set) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 score.

    Args:
        predicted: Set of predicted items
        ground_truth: Set of ground truth items

    Returns:
        Dictionary with 'precision', 'recall', 'f1' keys
    """
    if not predicted:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    if not ground_truth:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    tp = len(predicted & ground_truth)

    if tp == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    precision = tp / len(predicted)
    recall = tp / len(ground_truth)
    f1 = 2 * precision * recall / (precision + recall)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compute_shd(predicted_graph: nx.DiGraph,
                ground_truth_graph: nx.DiGraph) -> int:
    """
    Compute Structural Hamming Distance (SHD) between two graphs.

    Following the RQ1 benchmark implementation (metrics.py), SHD counts:
    - Edge present in G1 but missing in G2
    - Edge missing in G1 but present in G2
    - Edge undirected in G1 but directed in G2
    - Edge directed in G1 but undirected or reversed in G2

    Args:
        predicted_graph: Predicted causal graph
        ground_truth_graph: Ground truth causal graph

    Returns:
        SHD value (lower is better)
    """
    # Get all nodes from both graphs
    all_nodes = set(predicted_graph.nodes()) | set(ground_truth_graph.nodes())

    shd = 0

    for i, j in combinations(all_nodes, 2):
        # Helper functions
        def has_both_edges(g, x, y):
            return g.has_edge(x, y) and g.has_edge(y, x)

        def has_any_edge(g, x, y):
            return g.has_edge(x, y) or g.has_edge(y, x)

        def has_only_edge(g, x, y):
            return g.has_edge(x, y) and (not g.has_edge(y, x))

        def has_no_edge(g, x, y):
            return not (g.has_edge(x, y) or g.has_edge(y, x))

        # Edge present in GT, but missing in predicted
        if has_any_edge(ground_truth_graph, i, j) and has_no_edge(predicted_graph, i, j):
            shd += 1
        # Edge missing in GT, but present in predicted
        elif has_no_edge(ground_truth_graph, i, j) and has_any_edge(predicted_graph, i, j):
            shd += 1
        # Edge undirected in GT (both directions), but directed in predicted
        elif has_both_edges(ground_truth_graph, i, j) and (
            has_only_edge(predicted_graph, i, j) or has_only_edge(predicted_graph, j, i)
        ):
            shd += 1
        # Edge directed in GT, but undirected or reversed in predicted
        elif (has_only_edge(ground_truth_graph, i, j) and predicted_graph.has_edge(j, i)) or (
            has_only_edge(ground_truth_graph, j, i) and predicted_graph.has_edge(i, j)
        ):
            shd += 1

    return shd


def compute_edge_metrics(predicted_graph: nx.DiGraph,
                        ground_truth_graph: nx.DiGraph) -> Dict[str, float]:
    """
    Compute edge-level precision, recall, and F1 (following RQ1 benchmark).

    Args:
        predicted_graph: Predicted causal graph
        ground_truth_graph: Ground truth causal graph

    Returns:
        Dictionary with 'precision', 'recall', 'f1' keys
    """
    pred_edges = set(predicted_graph.edges())
    gt_edges = set(ground_truth_graph.edges())
    return compute_precision_recall_f1(pred_edges, gt_edges)


def compute_skeleton_metrics(predicted_graph: nx.DiGraph,
                            ground_truth_graph: nx.DiGraph) -> Dict[str, float]:
    """
    Compute metrics for the skeleton (undirected) graph (following RQ1 benchmark).

    Args:
        predicted_graph: Predicted causal graph
        ground_truth_graph: Ground truth causal graph

    Returns:
        Dictionary with skeleton edge metrics
    """
    # Convert to skeleton (undirected) by adding both directions
    # Following RQ1 benchmark: double the single edge
    pred_edges = {(u, v) for u, v in predicted_graph.edges()} | {(v, u) for u, v in predicted_graph.edges()}
    gt_edges = {(u, v) for u, v in ground_truth_graph.edges()} | {(v, u) for u, v in ground_truth_graph.edges()}

    return compute_precision_recall_f1(pred_edges, gt_edges)


def compute_node_metrics(predicted_graph: nx.DiGraph,
                        ground_truth_graph: nx.DiGraph) -> Dict[str, float]:
    """
    Compute node-level precision and recall.

    Args:
        predicted_graph: Predicted causal graph
        ground_truth_graph: Ground truth causal graph

    Returns:
        Dictionary with 'precision', 'recall', 'f1' keys
    """
    pred_nodes = set(predicted_graph.nodes())
    gt_nodes = set(ground_truth_graph.nodes())

    return compute_precision_recall_f1(pred_nodes, gt_nodes)
