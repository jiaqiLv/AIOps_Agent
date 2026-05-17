"""
Root Cause Analysis (RCA) engine.

This module implements the main engine for root cause analysis using time-shift
adjusted correlation and causal graph traversal.
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import logging

from .correlation_adjustment import compute_adjusted_correlations_all, filter_by_adjusted_correlation
from .graph_subtraction import extract_root_cause_subgraph, classify_subgraph_nodes

logger = logging.getLogger(__name__)


@dataclass
class RCAConfig:
    """Configuration parameters for RCA analysis."""
    # Correlation adjustment parameters
    max_shift: int = 1
    max_width: int = 2
    shift_penalty: float = 0.004
    smooth_penalty: float = 0.01
    min_similarity: float = 0.5

    # Graph parameters
    max_path_length: int = 10


@dataclass
class RCAResult:
    """Result of root cause analysis."""
    root_causes: List[str]
    subgraph: nx.DiGraph
    node_classifications: Dict[str, str]
    correlation_results: Dict[str, Tuple[float, float, float]]
    config: RCAConfig


class RCAEngine:
    """
    Root Cause Analysis Engine.

    Uses time-shift adjusted correlation and causal graph traversal to identify
    root causes from a symptom metric.

    Workflow:
    1. Compute adjusted correlation scores for all candidates
    2. Filter candidates based on scores
    3. Extract causal subgraph using graph subtraction
    4. Classify nodes (performance, root causes, intermediate)
    """

    def __init__(self, config: Optional[RCAConfig] = None):
        """
        Initialize the RCA Engine.

        Args:
            config: Configuration parameters
        """
        self.config = config or RCAConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    def analyze(self,
                data: pd.DataFrame,
                causal_graph: nx.DiGraph,
                performance_metric: str,
                node_levels: Optional[Dict[str, int]] = None,
                graph_refinement_knowledge: Optional[Dict[str, str]] = None) -> RCAResult:
        """
        Perform root cause analysis.

        Args:
            data: DataFrame with metrics as columns
            causal_graph: Causal graph (DiGraph)
            performance_metric: Name of the performance metric
            node_levels: Optional level assignments
            graph_refinement_knowledge: Optional correlation direction knowledge

        Returns:
            RCAResult with analysis results
        """
        self.logger.info(f"Starting RCA for performance metric: {performance_metric}")

        # Extract performance and candidate metrics
        if performance_metric not in data.columns:
            raise ValueError(f"Performance metric '{performance_metric}' not found in data")

        performance_ts = data[performance_metric].values
        candidate_metrics = {
            col: data[col].values
            for col in data.columns
            if col != performance_metric
        }

        # Step 1: Compute adjusted correlation scores
        self.logger.info("Computing adjusted correlation scores...")
        correlation_results = compute_adjusted_correlations_all(
            performance_ts,
            candidate_metrics,
            max_shift=self.config.max_shift,
            max_width=self.config.max_width,
            shift_penalty=self.config.shift_penalty,
            smooth_penalty=self.config.smooth_penalty
        )

        # Step 2: Filter candidates
        root_causes = filter_by_adjusted_correlation(
            correlation_results,
            min_similarity=self.config.min_similarity,
            graph_refinement_knowledge=graph_refinement_knowledge
        )

        self.logger.info(f"Found {len(root_causes)} candidate root causes")

        # Step 3: Extract causal subgraph
        correlation_scores = {name: data[0] for name, data in correlation_results.items()}

        subgraph = extract_root_cause_subgraph(
            causal_graph,
            performance_metric,
            root_causes,
            node_levels=node_levels,
            correlation_scores=correlation_scores,
            max_path_length=self.config.max_path_length
        )

        self.logger.info(f"Extracted subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")

        # Step 4: Filter root causes to only those in the subgraph
        # Keep only root causes that have a causal path to the performance metric
        root_cause_set = set(root_causes)
        node_classifications = classify_subgraph_nodes(
            subgraph,
            performance_metric,
            root_cause_set
        )

        # Get root causes that are actually in the subgraph
        root_causes_in_subgraph = [
            node for node, classification in node_classifications.items()
            if classification == 'root_cause'
        ]

        # Re-sort by correlation score to maintain ranking
        root_causes_in_subgraph = sorted(
            root_causes_in_subgraph,
            key=lambda x: correlation_scores.get(x, 0),
            reverse=True
        )

        filtered_count = len(root_causes) - len(root_causes_in_subgraph)
        if filtered_count > 0:
            self.logger.info(f"Filtered out {filtered_count} root causes without causal path to symptom")

        # Step 5: Create result
        result = RCAResult(
            root_causes=root_causes_in_subgraph,
            subgraph=subgraph,
            node_classifications=node_classifications,
            correlation_results=correlation_results,
            config=self.config
        )

        self.logger.info(f"RCA complete: {len(root_causes_in_subgraph)} root causes, {subgraph.number_of_nodes()} nodes in subgraph")

        return result

    def get_root_causes_sorted(self, result: RCAResult) -> List[Tuple[str, float]]:
        """Get root causes sorted by adjusted correlation score."""
        scores = [
            (name, result.correlation_results[name][0])
            for name in result.root_causes
            if name in result.correlation_results
        ]
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def get_intermediate_metrics(self, result: RCAResult) -> List[str]:
        """Get intermediate metrics (on causal paths but not root causes)."""
        return [
            name for name, classification in result.node_classifications.items()
            if classification == 'intermediate'
        ]

    def print_summary(self, result: RCAResult) -> None:
        """Print a summary of the RCA result."""
        print("=" * 60)
        print("RCA Analysis Summary")
        print("=" * 60)

        print(f"\nConfiguration:")
        print(f"  max_shift: {result.config.max_shift}")
        print(f"  max_width: {result.config.max_width}")
        print(f"  min_similarity: {result.config.min_similarity}")

        print(f"\nRoot Causes ({len(result.root_causes)}):")
        sorted_causes = self.get_root_causes_sorted(result)
        for i, (name, score) in enumerate(sorted_causes, 1):
            corr = result.correlation_results[name][1]
            print(f"  {i}. {name}")
            print(f"     Score: {score:.4f}, Correlation: {corr:.4f}")

        intermediate = self.get_intermediate_metrics(result)
        if intermediate:
            print(f"\nIntermediate Metrics ({len(intermediate)}):")
            for i, name in enumerate(intermediate, 1):
                print(f"  {i}. {name}")

        print(f"\nCausal Subgraph:")
        print(f"  Nodes: {result.subgraph.number_of_nodes()}")
        print(f"  Edges: {result.subgraph.number_of_edges()}")

        print(f"\nNode Classifications:")
        for classification in ['performance', 'root_cause', 'intermediate']:
            count = sum(1 for c in result.node_classifications.values() if c == classification)
            print(f"  {classification}: {count}")

        print("=" * 60)
