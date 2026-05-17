"""
Cascade orientation for iterative edge orientation in causal graphs.

This module implements a cascade orientation process that:
1. Takes PC algorithm's CausalGraph as input
2. Uses time-lag analysis to orient edges based on temporal precedence
3. Uses IGCI entropy to orient remaining edges
4. Calls causal-learn's meek() method after each edge orientation
5. Continues until all edges are oriented

Reference: Optimized PC orientation with time lag and IGCI.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.signal import correlate
import logging

try:
    from causallearn.graph.GeneralGraph import GeneralGraph
    from causallearn.graph.GraphClass import CausalGraph
    from causallearn.utils.PCUtils.Meek import meek
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CAUSAL_LEARN_AVAILABLE = False
    GeneralGraph = None
    CausalGraph = None
    meek = None

from src.orientation.orientation_tracker import OrientationTracker, OrientationMethod

logger = logging.getLogger(__name__)


@dataclass
class OrientationScore:
    """Score and direction for an edge orientation decision."""
    edge: Tuple[int, int]  # (node_idx_i, node_idx_j)
    direction: Tuple[int, int]  # (source_idx, target_idx)
    score: float
    method: str  # 'TIME_LAG' or 'IGCI'
    correlation: float = 0.0
    time_lag: int = 0
    igci_confidence: float = 0.0


class CascadeOrientator:
    """
    Cascade orientation for PC algorithm output.

    Strategy:
    1. Input: CausalGraph from PC algorithm (contains GeneralGraph.G)
    2. For remaining undirected edges:
       - Try time-lag orientation (high priority)
       - Try IGCI orientation (low priority)
    3. After each orientation, apply Meek rules via causal-learn's meek()
    4. Repeat until all edges are oriented
    """

    # Base score for time-lag orientation (gives priority over IGCI)
    TIME_LAG_BASE_SCORE = 10.0

    def __init__(self, correlation_threshold: float = 0.3,
                 igci_confidence_threshold: float = 0.01,
                 max_time_lag: int = 10,
                 max_iterations: int = 100):
        """
        Initialize the cascade orientator.

        Args:
            correlation_threshold: Minimum absolute cross-correlation for time-lag orientation
            igci_confidence_threshold: Minimum entropy difference for IGCI orientation
            max_time_lag: Maximum absolute time lag for valid orientation
            max_iterations: Maximum number of iterations
        """
        self.correlation_threshold = correlation_threshold
        self.igci_confidence_threshold = igci_confidence_threshold
        self.max_time_lag = max_time_lag
        self.max_iterations = max_iterations
        self.tracker = OrientationTracker()
        self.logger = logging.getLogger(self.__class__.__name__)

    def orient_graph(self, cg: CausalGraph,
                    data: pd.DataFrame,
                    node_names: List[str],
                    enable_time_lag: bool = True,
                    enable_igci: bool = True) -> CausalGraph:
        """
        Orient the partially directed graph using cascade orientation.

        Args:
            cg: CausalGraph from PC algorithm (contains GeneralGraph.G)
            data: Time series data for orientation analysis
            node_names: List of node names corresponding to graph indices
            enable_time_lag: Whether to apply time lag analysis
            enable_igci: Whether to apply IGCI analysis

        Returns:
            Oriented CausalGraph
        """
        if not CAUSAL_LEARN_AVAILABLE:
            raise ImportError("causal-learn library is required")

        self.logger.info("Starting cascade orientation")
        self.logger.info(f"Correlation threshold: {self.correlation_threshold}")
        self.logger.info(f"Max time lag: {self.max_time_lag}")
        self.logger.info(f"IGCI confidence threshold: {self.igci_confidence_threshold}")

        # Input CausalGraph already has Meek rules applied
        initial_undirected = self._count_undirected_edges(cg)
        self.logger.info(f"Input undirected edges: {initial_undirected}")

        # Phase 1: Compute orientation scores for ALL undirected edges
        undirected_edges = self._find_undirected_edges(cg)
        self.logger.info(f"Computing orientation scores for {len(undirected_edges)} undirected edges")

        all_scores = self._compute_orientation_scores(
            cg, undirected_edges, data, node_names,
            enable_time_lag, enable_igci
        )

        self.logger.info(f"Computed {len(all_scores)} orientation candidates")

        # Phase 2: Sort by confidence score (descending)
        all_scores.sort(key=lambda x: x.score, reverse=True)

        # Log distribution of methods
        time_lag_count = sum(1 for s in all_scores if s.method == 'TIME_LAG')
        igci_count = sum(1 for s in all_scores if s.method == 'IGCI')
        self.logger.info(f"Orientation candidates: {time_lag_count} TIME_LAG, {igci_count} IGCI")

        # Phase 3: Orient edges in order of confidence
        oriented_count = 0
        skipped_count = 0

        for score_info in all_scores:
            i, j = score_info.edge
            src_idx, dst_idx = score_info.direction

            # Check if edge is still undirected (may have been oriented by Meek rules)
            if not self._is_edge_undirected(cg, i, j):
                self.logger.debug(
                    f"Edge {node_names[i]}-{node_names[j]} already oriented by Meek rules, skipping"
                )
                skipped_count += 1
                continue

            # Apply orientation
            self._orient_edge_in_causal_graph(cg, src_idx, dst_idx)

            self.logger.debug(
                f"Oriented edge {node_names[src_idx]} -> {node_names[dst_idx]} "
                f"using {score_info.method} (confidence: {score_info.score:.4f})"
            )

            # Record orientation
            src_name = node_names[src_idx]
            dst_name = node_names[dst_idx]
            method = OrientationMethod.TIME_LAG if score_info.method == 'TIME_LAG' else OrientationMethod.IGCI
            self.tracker.record_orientation(
                edge=(src_name, dst_name),
                original_direction='undirected',
                final_direction=f'{src_name}->{dst_name}',
                method=method,
                method_detail=score_info.method,
                confidence=score_info.score
            )
            oriented_count += 1

            # Apply Meek rules to propagate orientation and maintain acyclicity
            cg = meek(cg)

        final_undirected = self._count_undirected_edges(cg)
        self.logger.info(
            f"Orientation complete: {oriented_count} edges oriented, "
            f"{skipped_count} skipped (already oriented by Meek), "
            f"{final_undirected} undirected edges remain"
        )

        return cg

    def _compute_orientation_scores(self, cg: CausalGraph,
                                    undirected_edges: List[Tuple[int, int]],
                                    data: pd.DataFrame,
                                    node_names: List[str],
                                    enable_time_lag: bool,
                                    enable_igci: bool) -> List[OrientationScore]:
        """
        Compute orientation scores for undirected edges.

        Returns list of OrientationScore objects sorted by score.
        """
        scores = []

        for i, j in undirected_edges:
            name_i = node_names[i]
            name_j = node_names[j]

            # Skip if data not available
            if name_i not in data.columns or name_j not in data.columns:
                continue

            x = data[name_i].values
            y = data[name_j].values

            # Track whether this edge has been oriented
            oriented = False

            # Try time-lag orientation first (high priority)
            if enable_time_lag:
                time_lag_result = self._compute_time_lag(x, y)
                max_corr = time_lag_result['max_correlation']

                # Use cross-correlation max value as threshold check
                if abs(max_corr) >= self.correlation_threshold:
                    direction = self._get_direction_from_time_lag(time_lag_result)

                    if direction is not None:
                        # High priority score for time-lag orientation
                        score = self.TIME_LAG_BASE_SCORE + abs(max_corr)

                        # direction: 0 means x->y (i->j), 1 means y->x (j->i)
                        if direction == 0:  # x causes y
                            src_idx, dst_idx = i, j
                        else:  # y causes x
                            src_idx, dst_idx = j, i

                        scores.append(OrientationScore(
                            edge=(i, j),
                            direction=(src_idx, dst_idx),
                            score=score,
                            method='TIME_LAG',
                            correlation=max_corr,
                            time_lag=time_lag_result['max_lag']
                        ))
                        oriented = True

            # Try IGCI orientation only if time-lag failed
            if not oriented and enable_igci:
                igci_result = self._compute_igci(x, y)
                confidence = igci_result['confidence']

                # Only orient if confidence exceeds threshold
                if confidence >= self.igci_confidence_threshold:
                    direction = igci_result['direction']

                    if direction == 0:  # x causes y
                        scores.append(OrientationScore(
                            edge=(i, j),
                            direction=(i, j),
                            score=confidence,
                            method='IGCI',
                            igci_confidence=confidence
                        ))
                    else:  # y causes x
                        scores.append(OrientationScore(
                            edge=(i, j),
                            direction=(j, i),
                            score=confidence,
                            method='IGCI',
                            igci_confidence=confidence
                        ))

        return scores

    def _compute_time_lag(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Compute time lag using normalized cross-correlation.

        Returns normalized cross-correlation coefficients in [-1, 1] range.
        Uses accurate normalization by dividing by effective overlap at each lag.
        """
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]

        if len(x_clean) < 10:
            return {'max_lag': 0, 'max_correlation': 0}

        # Normalize to zero mean, unit variance
        x_norm = (x_clean - np.mean(x_clean)) / (np.std(x_clean) + 1e-8)
        y_norm = (y_clean - np.mean(y_clean)) / (np.std(y_clean) + 1e-8)

        n = len(x_norm)

        # Compute cross-correlation
        cross_corr = correlate(y_norm, x_norm, mode='full')
        lags = np.arange(-n + 1, n)

        # Accurate normalization: divide by effective overlap at each lag
        # For lag k, the number of overlapping points is n - abs(k)
        overlaps = n - np.abs(lags)
        # Avoid division by zero
        overlaps = np.maximum(overlaps, 1)
        correlation = cross_corr / overlaps

        # Limit to reasonable lag range
        max_lag_limit = min(20, n // 4)
        lag_mask = (lags >= -max_lag_limit) & (lags <= max_lag_limit)
        lags = lags[lag_mask]
        correlation = correlation[lag_mask]

        # Find maximum absolute correlation
        max_idx = np.argmax(np.abs(correlation))
        max_lag = int(lags[max_idx])
        max_correlation = float(correlation[max_idx])

        return {'max_lag': max_lag, 'max_correlation': max_correlation}

    def _get_direction_from_time_lag(self, time_lag_result: Dict) -> Optional[int]:
        """
        Determine direction from time lag analysis.

        Returns 0 if x->y, 1 if y->x, None if undetermined.
        """
        max_lag = time_lag_result['max_lag']
        max_corr = time_lag_result['max_correlation']

        # Check if correlation is significant
        if abs(max_corr) < self.correlation_threshold:
            return None

        # Check if time lag is within acceptable range
        # Only use time lag for orientation if abs(lag) <= max_time_lag
        if abs(max_lag) > self.max_time_lag:
            return None

        # Positive lag means y leads x (y -> x)
        # Negative lag means x leads y (x -> y)
        if max_lag > 0:
            return 1  # y causes x
        elif max_lag < 0:
            return 0  # x causes y

        return None

    def _compute_igci(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Compute IGCI score for both directions.

        Returns: {'confidence': float, 'direction': int}
                 direction: 0 if x->y, 1 if y->x
        """
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]

        if len(x_clean) < 10:
            return {'confidence': 0.0, 'direction': 0}

        # Normalize to [0, 1]
        def normalize(v):
            v_min, v_max = v.min(), v.max()
            if v_max - v_min < 1e-10:
                return np.zeros_like(v)
            return (v - v_min) / (v_max - v_min)

        x_norm = normalize(x_clean)
        y_norm = normalize(y_clean)

        # Compute scores for both directions
        score_xy = self._compute_igci_unidirectional(x_norm, y_norm)
        score_yx = self._compute_igci_unidirectional(y_norm, x_norm)

        # Confidence is the absolute difference
        confidence = abs(float(score_xy - score_yx))

        # Lower score indicates more likely causal direction
        direction = 0 if score_xy < score_yx else 1

        return {'confidence': confidence, 'direction': direction,
                'score_xy': float(score_xy), 'score_yx': float(score_yx)}

    def _compute_igci_unidirectional(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute IGCI score for a single direction (lower = more likely causal).

        Uses dynamic bin selection based on sample size for more accurate entropy estimation.
        """
        # Sort by x
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]

        # Compute differences
        dx = np.diff(x_sorted)
        dy = np.diff(y_sorted)

        # Filter out zero or near-zero dx values
        mask = np.abs(dx) > 1e-10
        if np.sum(mask) < 2:
            return 0.0

        dx_filtered = dx[mask]
        dy_filtered = dy[mask]

        # Compute correlation-based score using entropy of ratios
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = dy_filtered / dx_filtered
            ratios = ratios[np.isfinite(ratios)]

            if len(ratios) < 2:
                return 0.0

            # Dynamic bin selection using Scott's rule: bins ~ n^(1/3)
            # Ensure at least 5 bins and at most min(50, n/2) bins
            n_ratios = len(ratios)
            n_bins = max(5, min(50, int(n_ratios ** (1/3) * 2)))

            # Entropy-based score with dynamic bins
            histogram, _ = np.histogram(ratios, bins=n_bins, density=True)
            histogram = histogram[histogram > 0]
            score = -np.sum(histogram * np.log(histogram + 1e-10))

        return float(score)

    def _count_undirected_edges(self, cg: CausalGraph) -> int:
        """Count the number of undirected edges in the graph."""
        graph_matrix = cg.G.graph
        count = 0
        n = cg.G.get_num_nodes()

        for i in range(n):
            for j in range(i + 1, n):  # Only count each pair once
                # Undirected edges are represented as -1, -1
                if graph_matrix[i, j] == -1 and graph_matrix[j, i] == -1:
                    count += 1

        return count

    def _find_undirected_edges(self, cg: CausalGraph) -> List[Tuple[int, int]]:
        """Find all undirected edges in the graph."""
        undirected = []
        graph_matrix = cg.G.graph
        n = cg.G.get_num_nodes()

        for i in range(n):
            for j in range(i + 1, n):
                if graph_matrix[i, j] == -1 and graph_matrix[j, i] == -1:
                    undirected.append((i, j))

        return undirected

    def _is_edge_undirected(self, cg: CausalGraph, i: int, j: int) -> bool:
        """Check if edge between i and j is undirected."""
        graph_matrix = cg.G.graph
        return graph_matrix[i, j] == -1 and graph_matrix[j, i] == -1

    def _orient_edge_in_causal_graph(self, cg: CausalGraph, src_idx: int, dst_idx: int) -> None:
        """
        Orient an undirected edge as src_idx -> dst_idx in the CausalGraph.

        Modifies the graph in place.
        """
        graph_matrix = cg.G.graph

        # Set directed edge: src_idx -> dst_idx
        # In causal-learn format: graph[dst, src] = 1, graph[src, dst] = -1
        graph_matrix[dst_idx, src_idx] = 1
        graph_matrix[src_idx, dst_idx] = -1

    def causal_graph_to_dataframe(self, cg: CausalGraph,
                                   node_names: List[str]) -> pd.DataFrame:
        """
        Convert a CausalGraph to a pandas DataFrame adjacency matrix.

        Args:
            cg: causal-learn CausalGraph object
            node_names: List of node names corresponding to graph indices

        Returns:
            DataFrame adjacency matrix where:
            - -1 indicates directed edge (i -> j)
            - 1 indicates undirected edge (i - j)
            - 0 indicates no edge between nodes
        """
        n = cg.G.get_num_nodes()
        adj_matrix = np.zeros((n, n), dtype=int)

        # Get the internal graph matrix
        graph_matrix = cg.G.graph

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                # In causal-learn format:
                # graph[j,i] = 1, graph[i,j] = -1 means i -> j (directed)
                # graph[i,j] = -1, graph[j,i] = -1 means i - j (undirected)
                val_ij = graph_matrix[i, j]
                val_ji = graph_matrix[j, i]

                if val_ij == -1 and val_ji == 1:
                    # i -> j (directed from i to j)
                    adj_matrix[i, j] = -1
                elif val_ij == -1 and val_ji == -1:
                    # i - j (undirected)
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1

        return pd.DataFrame(adj_matrix, index=node_names, columns=node_names)

    def get_tracker(self) -> OrientationTracker:
        """Get the orientation tracker."""
        return self.tracker
