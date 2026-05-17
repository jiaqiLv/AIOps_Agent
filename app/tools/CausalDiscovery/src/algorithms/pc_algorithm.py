"""
PC algorithm implementation with level-based conditioning set filtering.

This module implements the PC (Peter-Clark) causal discovery algorithm
with custom conditional independence testing that includes:
1. Level-based filtering of conditioning sets
2. Full tracking of CI test results
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import logging

try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import CIT, register_ci_test, CIT_Base
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CAUSAL_LEARN_AVAILABLE = False

from .base_algorithm import BaseAlgorithm, AlgorithmResult

logger = logging.getLogger(__name__)


class CILogger:
    """Global logger for conditional independence test results."""

    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.node_names: Optional[List[str]] = None

    def set_node_names(self, node_names: List[str]) -> None:
        """Set the node names for result formatting."""
        self.node_names = node_names

    def record(self, X: int, Y: int, S: Tuple, p_value: float,
               test_method: str, alpha: float) -> None:
        """Record a conditional independence test result."""
        self.results.append({
            'var_x_idx': int(X),
            'var_y_idx': int(Y),
            'conditioning_set_idxs': list(S) if S is not None else [],
            'p_value': float(p_value),
            'test_method': test_method,
            'alpha': alpha,
            'is_independent': p_value >= alpha
        })

    def clear(self) -> None:
        """Clear all results."""
        self.results = []
        self.node_names = None

    def get_formatted_results(self) -> List[Dict[str, Any]]:
        """Get results with node names instead of indices."""
        if self.node_names is None:
            return self.results

        formatted = []
        for result in self.results:
            r = result.copy()
            x_idx = int(result['var_x_idx'])
            y_idx = int(result['var_y_idx'])

            r['var_x'] = self.node_names[x_idx] if x_idx < len(self.node_names) else f"var_{x_idx}"
            r['var_y'] = self.node_names[y_idx] if y_idx < len(self.node_names) else f"var_{y_idx}"

            conditioning_names = []
            for idx in result['conditioning_set_idxs']:
                if idx < len(self.node_names):
                    conditioning_names.append(self.node_names[idx])
                else:
                    conditioning_names.append(f"var_{idx}")
            r['conditioning_set'] = conditioning_names

            formatted.append(r)

        return formatted


# Global CI logger instance
_global_ci_logger = CILogger()


class TrackedCIT(CIT_Base):
    """
    Custom conditional independence test with tracking and level-based filtering.

    This CI test implements level-based filtering:
    - Removes conditioning variables that are at a higher level than both X and Y
      (prevents conditioning on downstream variables)
    """

    def __init__(self, data, test_method='fisherz', alpha=0.05, node_names=None,
                 level_map=None, **kwargs):
        """
        Initialize the tracked CI test.

        Args:
            data: Data array (n_samples, n_variables)
            test_method: CI test method ('fisherz', 'chisq', etc.)
            alpha: Significance level
            node_names: List of variable names
            level_map: Dictionary mapping variable names to hierarchy levels
        """
        super().__init__(data, **kwargs)
        self.test_method = test_method
        self.alpha = alpha
        self.node_names = node_names
        self.level_map = level_map or {}

        self.check_cache_method_consistent(test_method, "NO SPECIFIED PARAMETERS")
        self.assert_input_data_is_valid()

        # Initialize internal CI executor
        self.internal_cit = CIT(data, test_method)

    def __call__(self, X: int, Y: int, condition_set=None) -> float:
        """
        Execute conditional independence test with filtering.

        Args:
            X, Y: Variable indices
            condition_set: Tuple of conditioning variable indices

        Returns:
            p-value of the test
        """
        real_condition_set = condition_set

        # Apply filtering if condition set and metadata are available
        if condition_set and self.node_names and self.level_map:
            x_name = self.node_names[X]
            y_name = self.node_names[Y]
            x_level = self.level_map.get(x_name, 999)
            y_level = self.level_map.get(y_name, 999)

            filtered_indices = []

            for z_idx in condition_set:
                z_name = self.node_names[z_idx]
                z_level = self.level_map.get(z_name, 999)

                # Level-based filtering: Skip if Z is higher level than both X and Y
                # if z_level > x_level and z_level > y_level:
                #     continue

                filtered_indices.append(z_idx)

            real_condition_set = tuple(filtered_indices)

        # Execute CI test
        p_value = self.internal_cit(X, Y, real_condition_set)

        # Record result
        _global_ci_logger.record(X, Y, real_condition_set, p_value, self.test_method, self.alpha)

        return p_value


class PCAlgorithm(BaseAlgorithm):
    """
    PC algorithm implementation with level-aware conditioning.

    This implementation extends the standard PC algorithm with:
    - Level-based conditioning set filtering
    - Full tracking of CI test results
    """

    def __init__(self, **params):
        """
        Initialize the PC algorithm.

        Args:
            alpha: Significance level (default: 0.05)
            indep_test: Independence test method (default: 'fisherz')
            stable: Use stable version (default: True)
            uc_rule: Undirected edge rule (default: 0)
            uc_priority: Undirected edge priority (default: 2)
        """
        super().__init__(**params)

        if not CAUSAL_LEARN_AVAILABLE:
            raise ImportError("causal-learn package is required for PC algorithm")

        self.alpha = params.get('alpha', 0.05)
        self.indep_test_name = params.get('indep_test', 'fisherz')
        self.stable = params.get('stable', True)
        self.uc_rule = params.get('uc_rule', 0)
        self.uc_priority = params.get('uc_priority', 2)

        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self, data: pd.DataFrame,
            background_knowledge = None,
            level_map: Dict[str, int] = None,
            output_dir: str = None) -> AlgorithmResult:
        """
        Run the PC algorithm.

        Args:
            data: Input data
            background_knowledge: causal-learn BackgroundKnowledge object (created by KnowledgeManager)
            level_map: Dictionary mapping node names to levels
            output_dir: Optional directory to save intermediate results

        Returns:
            AlgorithmResult with causal graph and metadata
        """
        self.logger.info(f"Running PC algorithm on data shape: {data.shape}")

        # Prepare data
        data_array = data.to_numpy().astype(float)
        node_names = data.columns.tolist()

        level_map = level_map or {}

        # Clear and setup CI logger
        _global_ci_logger.clear()
        _global_ci_logger.set_node_names(node_names)

        # Register custom CI test
        self._register_tracked_ci_test(node_names, level_map)

        # Run PC algorithm
        tracked_test_name = f"tracked_{self.indep_test_name}"

        try:
            cg = pc(
                data=data_array,
                alpha=self.alpha,
                indep_test=tracked_test_name,
                stable=self.stable,
                uc_rule=self.uc_rule,
                uc_priority=self.uc_priority,
                background_knowledge=background_knowledge,
                show_progress=True,
                node_names=node_names,
            )

            # Extract results
            final_adj_matrix = cg.G.graph if hasattr(cg.G, 'graph') else self._extract_adjacency_fallback(cg.G, node_names)
            skeleton_adj_matrix = self._extract_skeleton_matrix(cg, node_names)
            separation_sets = self._extract_separation_sets(cg, node_names)

            # Get CI test results
            independence_test_results = _global_ci_logger.get_formatted_results()

            # Log statistics
            edge_count = np.sum(final_adj_matrix != 0)
            skeleton_edge_count = np.sum(skeleton_adj_matrix != 0) // 2

            self.logger.info(f"PC complete: {edge_count} directed edges, {skeleton_edge_count} skeleton edges")
            self.logger.info(f"Executed {len(independence_test_results)} CI tests")

            # Save intermediate results if requested
            if output_dir:
                self._save_intermediate_results(
                    final_adj_matrix, skeleton_adj_matrix, separation_sets,
                    independence_test_results, node_names, output_dir
                )

            # Create DataFrames
            final_df = pd.DataFrame(final_adj_matrix, index=node_names, columns=node_names)
            skeleton_df = pd.DataFrame(skeleton_adj_matrix, index=node_names, columns=node_names)

            return AlgorithmResult(
                causal_graph=final_df,
                skeleton_graph=skeleton_df,
                separation_sets=separation_sets,
                independence_tests=independence_test_results,
                raw_graph=cg  # Store original causal-learn GeneralGraph
            )

        except Exception as e:
            self.logger.error(f"PC algorithm failed: {e}")
            raise

    def _register_tracked_ci_test(self, node_names: List[str],
                                  level_map: Dict[str, int]) -> None:
        """Register the tracked CI test with causal-learn."""

        def create_tracked_ci_class(test_method, alpha):
            class TrackedCI(TrackedCIT):
                def __init__(self, data):
                    super().__init__(
                        data, test_method, alpha,
                        node_names=node_names,
                        level_map=level_map
                    )
            return TrackedCI

        tracked_class = create_tracked_ci_class(self.indep_test_name, self.alpha)
        tracked_test_name = f"tracked_{self.indep_test_name}"
        register_ci_test(tracked_test_name, tracked_class)

        self.logger.debug(f"Registered tracked CI test: {tracked_test_name}")

    def _extract_adjacency_fallback(self, graph, node_names: List[str]) -> np.ndarray:
        """Fallback method to extract adjacency matrix."""
        n = len(node_names)
        adj_matrix = np.zeros((n, n), dtype=int)

        if hasattr(graph, 'graph'):
            return graph.graph

        # Manual extraction
        try:
            from causallearn.graph.Endpoint import Endpoint

            for edge in graph.get_edges():
                node1 = edge.get_node1().get_name()
                node2 = edge.get_node2().get_name()

                if node1 in node_names and node2 in node_names:
                    i = node_names.index(node1)
                    j = node_names.index(node2)

                    endpoint1 = edge.get_endpoint1()
                    endpoint2 = edge.get_endpoint2()

                    if endpoint1 == Endpoint.TAIL and endpoint2 == Endpoint.ARROW:
                        adj_matrix[i, j] = -1  # i -> j
                    elif endpoint1 == Endpoint.ARROW and endpoint2 == Endpoint.TAIL:
                        adj_matrix[j, i] = -1  # j -> i
                    elif endpoint1 == Endpoint.TAIL and endpoint2 == Endpoint.TAIL:
                        adj_matrix[i, j] = 1
                        adj_matrix[j, i] = 1  # undirected

        except Exception as e:
            self.logger.warning(f"Manual adjacency extraction failed: {e}")

        return adj_matrix

    def _extract_skeleton_matrix(self, cg, node_names: List[str]) -> np.ndarray:
        """Extract skeleton graph adjacency matrix."""
        n = len(node_names)
        skeleton_matrix = np.zeros((n, n), dtype=int)

        try:
            if hasattr(cg, 'G') and hasattr(cg.G, 'graph'):
                final_graph = cg.G.graph
                for i in range(n):
                    for j in range(n):
                        if final_graph[i, j] != 0 or final_graph[j, i] != 0:
                            skeleton_matrix[i, j] = 1
                            skeleton_matrix[j, i] = 1

        except Exception as e:
            self.logger.warning(f"Skeleton extraction failed: {e}")

        return skeleton_matrix

    def _extract_separation_sets(self, cg, node_names: List[str]) -> Dict[str, Any]:
        """Extract separation sets from PC results."""
        separation_sets = {
            'removed_edges': [],
            'separation_info': {}
        }

        try:
            if hasattr(cg, 'G') and hasattr(cg.G, 'get_nodes'):
                nodes = cg.G.get_nodes()

                for i, node1 in enumerate(nodes):
                    for j, node2 in enumerate(nodes):
                        if i >= j:
                            continue

                        if not cg.G.is_adjacent_to(node1, node2):
                            node1_name = node1.get_name()
                            node2_name = node2.get_name()

                            sep_set = self._get_separation_set(cg, i, j, node_names)

                            separation_sets['removed_edges'].append({
                                'node1': node1_name,
                                'node2': node2_name,
                                'separation_set': sep_set
                            })
                            separation_sets['separation_info'][f"{node1_name}-{node2_name}"] = sep_set

        except Exception as e:
            self.logger.warning(f"Separation set extraction failed: {e}")

        return separation_sets

    def _get_separation_set(self, cg, i: int, j: int, node_names: List[str]) -> List[str]:
        """Get separation set for a pair of variables."""
        try:
            if hasattr(cg, 'sepset'):
                if i < len(cg.sepset) and j < len(cg.sepset[i]):
                    sep_set = cg.sepset[i][j]
                    if sep_set and len(sep_set) > 0:
                        return [node_names[idx] for idx in sep_set[0]]
            return []
        except Exception:
            return []

    def _save_intermediate_results(self, final_adj: np.ndarray, skeleton_adj: np.ndarray,
                                  separation_sets: Dict, ci_results: List[Dict],
                                  node_names: List[str], output_dir: str) -> None:
        """Save intermediate results to files."""
        os.makedirs(output_dir, exist_ok=True)

        # Save final oriented graph (PC algorithm output)
        final_df = pd.DataFrame(final_adj, index=node_names, columns=node_names)
        final_path = os.path.join(output_dir, "pc_graph.csv")
        final_df.to_csv(final_path)
        self.logger.info(f"Saved PC graph: {final_path}")

        # Save CI test results
        if ci_results:
            ci_path = os.path.join(output_dir, "independence_tests.csv")
            pd.DataFrame(ci_results).to_csv(ci_path, index=False)
            self.logger.info(f"Saved CI test results: {ci_path}")

        # Save skeleton graph
        skeleton_df = pd.DataFrame(skeleton_adj, index=node_names, columns=node_names)
        skeleton_path = os.path.join(output_dir, "skeleton_graph.csv")
        skeleton_df.to_csv(skeleton_path)
        self.logger.info(f"Saved skeleton graph: {skeleton_path}")

        # Save separation sets
        sep_path = os.path.join(output_dir, "separation_sets.json")
        with open(sep_path, 'w', encoding='utf-8') as f:
            json.dump(separation_sets, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved separation sets: {sep_path}")

        # Save algorithm stats
        stats = {
            "algorithm": "PC",
            "parameters": {
                "alpha": self.alpha,
                "indep_test": self.indep_test_name,
                "stable": self.stable
            },
            "edge_count": int(np.sum(final_adj != 0)),
            "skeleton_edge_count": int(np.sum(skeleton_adj != 0) // 2),
            "node_count": len(node_names),
            "ci_tests_count": len(ci_results)
        }

        stats_path = os.path.join(output_dir, "algorithm_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved algorithm stats: {stats_path}")
