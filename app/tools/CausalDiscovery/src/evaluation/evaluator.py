"""
Main evaluation engine for causal discovery and RCA.

This module provides a unified interface for evaluating both
causal discovery results and root cause analysis results.

Supports RADICE dataset format with separate edges.txt (full causal graph)
and subgraph.txt (fault propagation subgraph) for evaluation.
"""

import json
import networkx as nx
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

from .graph_evaluator import GraphEvaluator
from .rca_evaluator import RCAEvaluator

logger = logging.getLogger(__name__)


class DatasetEvaluator:
    """
    Dataset-level evaluator that aggregates results across multiple samples.

    For RADICE datasets, this computes average metrics across all artificialResults_* samples.
    """

    def __init__(self):
        """Initialize the dataset evaluator."""
        self.sample_results: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(self.__class__.__name__)

    def add_sample_result(self, sample_result: Dict[str, Any]) -> None:
        """
        Add evaluation result for a single sample.

        Args:
            sample_result: Dictionary with 'graph' and/or 'subgraph' evaluation results
        """
        self.sample_results.append(sample_result)

    def compute_aggregate_metrics(self) -> Dict[str, Any]:
        """
        Compute aggregate (average) metrics across all samples.

        Returns:
            Dictionary with average metrics for:
            - graph: F1, F1-Skeleton, SHD
            - subgraph: F1, F1-Skeleton, SHD (if available)
        """
        if not self.sample_results:
            return {}

        aggregate = {}

        # Aggregate graph metrics
        graph_metrics = ['f1', 'precision', 'recall', 'skeleton_f1', 'skeleton_precision',
                        'skeleton_recall', 'shd']
        for metric in graph_metrics:
            values = []
            for result in self.sample_results:
                if 'graph' in result and metric in result['graph']:
                    values.append(result['graph'][metric])

            if values:
                aggregate[f'graph_{metric}'] = sum(values) / len(values)
                # For SHD, also report floor value (following RQ1 benchmark format)
                if metric == 'shd':
                    aggregate[f'graph_shd_floor'] = int(sum(values) / len(values))

        # Aggregate subgraph metrics (fault propagation subgraph)
        subgraph_metrics = ['f1', 'precision', 'recall', 'skeleton_f1', 'skeleton_precision',
                           'skeleton_recall', 'shd']
        for metric in subgraph_metrics:
            values = []
            for result in self.sample_results:
                if 'subgraph' in result and metric in result['subgraph']:
                    values.append(result['subgraph'][metric])

            if values:
                aggregate[f'subgraph_{metric}'] = sum(values) / len(values)
                if metric == 'shd':
                    aggregate[f'subgraph_shd_floor'] = int(sum(values) / len(values))

        # Add sample count
        aggregate['num_samples'] = len(self.sample_results)

        # Aggregate RCA metrics (following RQ2)
        # P = mean of per-case precisions
        # R = #samples_with_correct_root_cause / total_samples
        rca_precisions = []
        rca_recalls = []  # Case-level recalls
        rca_acc_at_1 = []
        rca_acc_at_2 = []
        rca_acc_at_3 = []
        rca_acc_at_4 = []
        rca_acc_at_5 = []

        for result in self.sample_results:
            if 'rca' in result:
                rca = result['rca']
                rca_precisions.append(rca.get('precision', 0.0))
                rca_recalls.append(rca.get('recall', 0.0))
                rca_acc_at_1.append(rca.get('acc@1', 0.0))
                rca_acc_at_2.append(rca.get('acc@2', 0.0))
                rca_acc_at_3.append(rca.get('acc@3', 0.0))
                rca_acc_at_4.append(rca.get('acc@4', 0.0))
                rca_acc_at_5.append(rca.get('acc@5', 0.0))

        if rca_precisions:
            aggregate['rca_precision'] = sum(rca_precisions) / len(rca_precisions)
            aggregate['rca_recall'] = sum(rca_recalls) / len(rca_recalls)
            aggregate['rca_acc@1'] = sum(rca_acc_at_1) / len(rca_acc_at_1)
            aggregate['rca_acc@2'] = sum(rca_acc_at_2) / len(rca_acc_at_2)
            aggregate['rca_acc@3'] = sum(rca_acc_at_3) / len(rca_acc_at_3)
            aggregate['rca_acc@4'] = sum(rca_acc_at_4) / len(rca_acc_at_4)
            aggregate['rca_acc@5'] = sum(rca_acc_at_5) / len(rca_acc_at_5)
            # Compute avg@5 = average of acc@1 to acc@5
            aggregate['rca_avg@5'] = (aggregate['rca_acc@1'] +
                                   aggregate['rca_acc@2'] +
                                   aggregate['rca_acc@3'] +
                                   aggregate['rca_acc@4'] +
                                   aggregate['rca_acc@5']) / 5

        self.logger.info(f"Aggregate metrics computed from {len(self.sample_results)} samples")

        return aggregate

    def print_summary(self) -> None:
        """Print summary of aggregate metrics in RQ1 format."""
        aggregate = self.compute_aggregate_metrics()

        if not aggregate:
            self.logger.warning("No aggregate metrics to display")
            return

        print("=" * 60)
        print("Aggregate Evaluation Results")
        print("=" * 60)

        # Graph metrics (following RQ1 output format)
        if 'graph_f1' in aggregate:
            print(f"Graph F1:   {aggregate['graph_f1']:.2f}")
        if 'graph_skeleton_f1' in aggregate:
            print(f"Graph F1-S: {aggregate['graph_skeleton_f1']:.2f}")
        if 'graph_shd_floor' in aggregate:
            print(f"Graph SHD:  {aggregate['graph_shd_floor']}")
        elif 'graph_shd' in aggregate:
            print(f"Graph SHD:  {aggregate['graph_shd']:.1f}")

        # Subgraph metrics (fault propagation)
        if 'subgraph_f1' in aggregate:
            print(f"Subgraph F1:   {aggregate['subgraph_f1']:.2f}")
        if 'subgraph_skeleton_f1' in aggregate:
            print(f"Subgraph F1-S: {aggregate['subgraph_skeleton_f1']:.2f}")
        if 'subgraph_shd_floor' in aggregate:
            print(f"Subgraph SHD:  {aggregate['subgraph_shd_floor']}")

        # RCA metrics (following RQ2 format)
        if 'rca_precision' in aggregate:
            print(f"RCA P:       {aggregate['rca_precision']:.3f}")
        if 'rca_recall' in aggregate:
            print(f"RCA R:       {aggregate['rca_recall']:.3f}")
        if 'rca_avg@5' in aggregate:
            print(f"RCA Avg@5:   {aggregate['rca_avg@5']:.3f}")

        print("=" * 60)


class EvaluationEngine:
    """
    Unified evaluation engine for causal discovery and RCA.

    Evaluates:
    - Causal graph structure against ground truth (edges.txt for RADICE)
    - Fault propagation subgraph against ground truth (subgraph.txt for RADICE)
    - Root cause localization against ground truth
    """

    def __init__(self):
        """Initialize the evaluation engine."""
        self.graph_evaluator = GraphEvaluator()
        self.rca_evaluator = RCAEvaluator()
        self.dataset_evaluator = DatasetEvaluator()
        self.logger = logging.getLogger(self.__class__.__name__)

    def evaluate(self,
                causal_graph: Optional[nx.DiGraph] = None,
                rca_subgraph: Optional[nx.DiGraph] = None,
                rca_results: Optional[Dict[str, Any]] = None,
                ground_truth_path: Optional[str] = None,
                ground_truth_dict: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run full evaluation.

        Args:
            causal_graph: Predicted full causal graph
            rca_subgraph: Predicted fault propagation subgraph (from RCA)
            rca_results: RCA results dict with 'root_causes' key
            ground_truth_path: Path to ground truth file (for RADICE: folder with edges.txt, subgraph.txt)
            ground_truth_dict: Direct ground truth dict

        Returns:
            Dictionary with 'graph' and/or 'subgraph' and/or 'rca' evaluation results
        """
        # Load ground truth
        if ground_truth_dict is None:
            if ground_truth_path:
                ground_truth_dict = self._load_ground_truth(ground_truth_path)
            else:
                self.logger.warning("No ground truth provided, returning empty evaluation")
                return {}

        results = {}

        # Evaluate full causal graph (against edges.txt for RADICE)
        if causal_graph is not None and ground_truth_dict.get('edges'):
            gt_graph = self._build_graph_from_ground_truth(ground_truth_dict)
            results['graph'] = self.graph_evaluator.evaluate(causal_graph, gt_graph)

        # Evaluate fault propagation subgraph (against subgraph.txt for RADICE)
        # Even if rca_subgraph is None/empty, evaluate to show "not found" metrics
        if ground_truth_dict.get('subgraph_edges'):
            gt_subgraph = self._build_graph_from_edges(ground_truth_dict['subgraph_edges'])

            # If no subgraph provided, create empty graph for evaluation
            if rca_subgraph is None:
                rca_subgraph = nx.DiGraph()

            results['subgraph'] = self.graph_evaluator.evaluate(rca_subgraph, gt_subgraph)

        # Evaluate root causes
        # Even if root_causes is empty, evaluate to show "not found" metrics
        if ground_truth_dict.get('root_causes'):
            gt_causes = ground_truth_dict['root_causes']

            # Handle case where root_causes is a dict with names
            if isinstance(gt_causes, dict):
                gt_causes = list(gt_causes.keys())

            # Get predicted causes (empty list if rca_results is None or root_causes is empty)
            predicted_causes = []
            if rca_results is not None:
                predicted_causes = rca_results.get('root_causes', [])
                # Ensure it's a list
                if predicted_causes is None:
                    predicted_causes = []

            results['rca'] = self.rca_evaluator.evaluate(
                predicted_causes,
                gt_causes
            )

            # Log if no root causes found
            if not predicted_causes:
                self.logger.info("No root causes found - evaluating empty prediction")

        # Store results for dataset-level aggregation
        self.dataset_evaluator.add_sample_result(results)

        self.logger.info("Evaluation complete")

        return results

    def _load_ground_truth(self, path: str) -> Dict:
        """
        Load ground truth from file or directory.

        For RADICE: path should point to the sample folder (e.g., artificialResults_0/)
                   which contains edges.txt, subgraph.txt, root_cause.txt

        Returns:
            Dictionary with ground truth data
        """
        path = Path(path)

        # If path is a directory (RADICE sample folder)
        if path.is_dir():
            return self._load_radice_sample(path)

        # If path is a file
        if path.suffix == '.txt':
            return self._load_gt_txt_format(path)

        # Default JSON format
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_radice_sample(self, sample_path: Path) -> Dict:
        """
        Load RADICE sample ground truth data.

        Args:
            sample_path: Path to RADICE sample folder (e.g., artificialResults_0/)

        Returns:
            Dictionary with 'edges', 'subgraph_edges', and 'root_causes'
        """
        gt = {}

        # Load edges.txt (full causal graph)
        edges_path = sample_path / 'edges.txt'
        if edges_path.exists():
            gt['edges'] = self._load_edges_file(edges_path)
            self.logger.debug(f"Loaded edges.txt: {len(gt['edges'])} edges")

        # Load subgraph.txt (fault propagation subgraph)
        subgraph_path = sample_path / 'subgraph.txt'
        if subgraph_path.exists():
            gt['subgraph_edges'] = self._load_edges_file(subgraph_path)
            self.logger.debug(f"Loaded subgraph.txt: {len(gt['subgraph_edges'])} edges")

        # Load root_cause.txt
        rc_path = sample_path / 'root_cause.txt'
        if rc_path.exists():
            root_causes, symptoms = self._load_root_cause_file(rc_path)
            gt['root_causes'] = root_causes
            gt['symptoms'] = symptoms
            self.logger.debug(f"Loaded root_cause.txt: root_causes={root_causes}, symptoms={symptoms}")

        return gt

    def _load_edges_file(self, edges_path: Path) -> List[List[str]]:
        """Load edges from edges.txt or subgraph.txt."""
        edges = []
        with open(edges_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    edges.append([str(parts[0]), str(parts[1])])
        return edges

    def _load_root_cause_file(self, rc_path: Path) -> tuple:
        """
        Load root causes and symptoms from root_cause.txt.

        Format: "<root_cause_id> <symptom_id>" (single line)
        Example: "6 9" (6 = root cause, 9 = symptom)

        Returns:
            Tuple of (root_causes, symptoms) as lists
        """
        with open(rc_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        parts = content.split()
        root_causes = []
        symptoms = []

        if len(parts) >= 1:
            root_causes = [str(parts[0])]
        if len(parts) >= 2:
            symptoms = [str(parts[1])]

        return root_causes, symptoms

    def _load_gt_txt_format(self, path: Path) -> Dict:
        """
        Load ground truth from custom gt.txt format.

        Returns:
            Dictionary with 'nodes', 'edges', and 'levels'
        """
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Parse first line: num_nodes num_edges
        parts = lines[0].strip().split()
        num_nodes = int(parts[0])

        # Parse second line: level assignments
        level_assignments = lines[1].strip().split()
        levels = {}
        nodes = set()

        for assignment in level_assignments:
            if ':' in assignment:
                node_id, level = assignment.split(':')
                nodes.add(node_id)
                levels[node_id] = int(level)

        node_list = sorted(nodes, key=lambda x: int(x) if x.isdigit() else x)

        # Parse edges
        edges = []
        for line in lines[3:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                edges.append([parts[0], parts[1]])

        return {
            'nodes': node_list,
            'edges': edges,
            'levels': levels,
            'root_causes': [node for node, level in levels.items() if level == 0]
        }

    def _build_graph_from_ground_truth(self, gt_dict: Dict) -> nx.DiGraph:
        """Build NetworkX graph from ground truth dict."""
        return self._build_graph_from_edges(gt_dict.get('edges', []))

    def _build_graph_from_edges(self, edges: List) -> nx.DiGraph:
        """Build NetworkX graph from edge list."""
        graph = nx.DiGraph()

        for edge in edges:
            if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                graph.add_edge(str(edge[0]), str(edge[1]))

        return graph
