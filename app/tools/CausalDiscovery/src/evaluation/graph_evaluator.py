"""
Graph evaluator for causal graphs.

This module evaluates predicted causal graphs against ground truth graphs,
following the RQ1 benchmark implementation.
"""

import networkx as nx
from typing import Dict, Any
import logging

from .metrics import (
    compute_shd,
    compute_edge_metrics,
    compute_skeleton_metrics
)

logger = logging.getLogger(__name__)


class GraphEvaluator:
    """
    Evaluator for causal graphs.

    Evaluates predicted causal graphs against ground truth.
    Output format matches RQ1 benchmark: F1, F1-Skeleton, SHD
    """

    def __init__(self):
        """Initialize the graph evaluator."""
        self.logger = logging.getLogger(self.__class__.__name__)

    def evaluate(self,
                predicted_graph: nx.DiGraph,
                ground_truth_graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Evaluate a predicted causal graph.

        Args:
            predicted_graph: Predicted causal graph
            ground_truth_graph: Ground truth causal graph

        Returns:
            Dictionary with evaluation metrics (matching RQ1 format):
            - f1: F1 score for directed edges
            - precision: Precision for directed edges
            - recall: Recall for directed edges
            - skeleton_f1: F1 score for skeleton (undirected edges)
            - skeleton_precision: Precision for skeleton
            - skeleton_recall: Recall for skeleton
            - shd: Structural Hamming Distance
        """
        results = {}

        # Edge metrics (following RQ1 F1 computation)
        edge_metrics = compute_edge_metrics(predicted_graph, ground_truth_graph)
        results['precision'] = edge_metrics['precision']
        results['recall'] = edge_metrics['recall']
        results['f1'] = edge_metrics['f1']

        # Skeleton metrics (following RQ1 F1-Skeleton computation)
        skeleton_metrics = compute_skeleton_metrics(predicted_graph, ground_truth_graph)
        results['skeleton_precision'] = skeleton_metrics['precision']
        results['skeleton_recall'] = skeleton_metrics['recall']
        results['skeleton_f1'] = skeleton_metrics['f1']

        # Structural Hamming Distance (following RQ1 SHD computation)
        results['shd'] = compute_shd(predicted_graph, ground_truth_graph)

        # Extra stats (for debugging)
        results['predicted_edge_count'] = predicted_graph.number_of_edges()
        results['ground_truth_edge_count'] = ground_truth_graph.number_of_edges()
        results['predicted_node_count'] = predicted_graph.number_of_nodes()
        results['ground_truth_node_count'] = ground_truth_graph.number_of_nodes()

        self.logger.info(
            f"Graph Evaluation: F1={results['f1']:.3f}, F1-S={results['skeleton_f1']:.3f}, SHD={results['shd']}"
        )

        return results
