"""
Base algorithm interface for causal discovery.

This module defines the abstract interface that all causal discovery algorithms must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np


class AlgorithmResult:
    """
    Result of causal discovery algorithm.

    Attributes:
        causal_graph: Adjacency matrix of the causal graph (as DataFrame)
        skeleton_graph: Adjacency matrix of the skeleton (undirected) graph
        separation_sets: Dictionary of separation sets for non-adjacent pairs
        independence_tests: List of conditional independence test results
        edge_count: Number of directed edges in the causal graph
        skeleton_edge_count: Number of edges in the skeleton graph
        raw_graph: Original causal-learn GeneralGraph object (if available)
    """

    def __init__(self, causal_graph: pd.DataFrame,
                 skeleton_graph: pd.DataFrame,
                 separation_sets: Dict[str, Any] = None,
                 independence_tests: List[Dict[str, Any]] = None,
                 raw_graph: Any = None):
        self.causal_graph = causal_graph
        self.skeleton_graph = skeleton_graph
        self.separation_sets = separation_sets or {}
        self.independence_tests = independence_tests or []
        self.raw_graph = raw_graph  # Store original causal-learn graph
        self.edge_count = int(np.sum(causal_graph.values != 0))
        self.skeleton_edge_count = int(np.sum(skeleton_graph.values != 0) // 2)

    def get_edges(self) -> List[Tuple[str, str]]:
        """Get list of directed edges as (source, target) tuples."""
        edges = []
        for i in range(len(self.causal_graph)):
            for j in range(len(self.causal_graph.columns)):
                if self.causal_graph.iloc[i, j] != 0:
                    source = self.causal_graph.index[i]
                    target = self.causal_graph.columns[j]
                    edges.append((source, target))
        return edges

    def get_nodes(self) -> List[str]:
        """Get list of node names."""
        return self.causal_graph.index.tolist()


class BaseAlgorithm(ABC):
    """
    Abstract base class for causal discovery algorithms.

    All algorithms must inherit from this class and implement the run() method.
    """

    def __init__(self, **params):
        """
        Initialize the algorithm with parameters.

        Args:
            **params: Algorithm-specific parameters
        """
        self.params = params

    @abstractmethod
    def run(self, data: pd.DataFrame,
            background_knowledge = None,
            level_map: Dict[str, int] = None,
            **kwargs) -> AlgorithmResult:
        """
        Run the causal discovery algorithm.

        Args:
            data: Input data (metrics as columns, time as rows)
            background_knowledge: causal-learn BackgroundKnowledge object (optional)
            level_map: Dictionary mapping node names to hierarchy levels
            **kwargs: Additional algorithm-specific arguments (e.g., output_dir)

        Returns:
            AlgorithmResult containing the discovered causal graph
        """
        pass

    def get_param(self, key: str, default: Any = None) -> Any:
        """Get a parameter value."""
        return self.params.get(key, default)

    def set_param(self, key: str, value: Any) -> None:
        """Set a parameter value."""
        self.params[key] = value
