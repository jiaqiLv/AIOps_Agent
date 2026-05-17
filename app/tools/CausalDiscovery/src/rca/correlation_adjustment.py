"""
Adjusted correlation score computation for RADICE.

This module implements Algorithm 2 from the RADICE paper for computing
adjusted correlation scores that account for noise and time shifts.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize time series to zero mean and unit variance."""
    mean = np.mean(x)
    std = np.std(x)
    if std < 1e-10:  # Use epsilon for floating point comparison
        return x - mean
    return (x - mean) / std


def smooth(x: np.ndarray, window_size: int) -> np.ndarray:
    """Apply moving average smoothing using uniform kernel."""
    if window_size <= 1:
        return x.copy()

    # Use uniform filter for efficiency (faster than manual convolution)
    from scipy.ndimage import uniform_filter1d
    smoothed = uniform_filter1d(x.astype(float), size=window_size, mode='nearest')
    return smoothed


def shift(x: np.ndarray, shift: int) -> np.ndarray:
    """Shift time series by specified steps using np.roll."""
    if shift == 0:
        return x.copy()

    # np.roll is more efficient than manual slicing
    x_shifted = np.roll(x, shift)

    # Zero out the wrapped portion
    if shift > 0:
        x_shifted[:shift] = 0
    else:
        x_shifted[shift:] = 0

    return x_shifted


def compute_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    # Vectorized masking for NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y))

    if np.sum(valid_mask) < 2:
        return 0.0

    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    # Use np.corrcoef with flat arrays
    return float(np.corrcoef(x_valid, y_valid)[0, 1])


def compute_adjusted_correlation(
    performance_metric: np.ndarray,
    candidate_metric: np.ndarray,
    max_shift: int = 1,
    max_width: int = 2,
    shift_penalty: float = 0.004,
    smooth_penalty: float = 0.01
) -> Tuple[float, float, float]:
    """
    Compute adjusted correlation score (Algorithm 2 from RADICE paper).

    Args:
        performance_metric: Performance metric time series
        candidate_metric: Candidate root cause time series
        max_shift: Maximum time shift to consider
        max_width: Maximum smoothing window size
        shift_penalty: Penalty per shift step
        smooth_penalty: Penalty per smoothing increment

    Returns:
        Tuple of (score, correlation, penalty)
    """
    x_norm = normalize(performance_metric)
    c_norm = normalize(candidate_metric)

    best_score = -np.inf
    best_corr = 0.0
    best_penalty = 0.0

    # Iterate over smoothing windows
    for w in range(1, max_width + 1):
        x_smooth = smooth(x_norm, w)
        c_smooth = smooth(c_norm, w)
        smooth_penalty_value = smooth_penalty * (w - 1)

        # Iterate over time shifts
        for s in range(max_shift + 1):
            c_shifted = shift(c_smooth, -s)
            corr = compute_correlation(x_smooth, c_shifted)
            penalty = smooth_penalty_value + shift_penalty * s
            score = abs(corr) - penalty

            # Track best result (higher score, lower penalty as tiebreaker)
            if score > best_score or (score == best_score and penalty < best_penalty):
                best_score = score
                best_corr = corr
                best_penalty = penalty

    return best_score if best_score > -np.inf else 0.0, best_corr, best_penalty


def compute_adjusted_correlations_all(
    performance_metric: np.ndarray,
    candidate_metrics: Dict[str, np.ndarray],
    max_shift: int = 1,
    max_width: int = 2,
    shift_penalty: float = 0.004,
    smooth_penalty: float = 0.01
) -> Dict[str, Tuple[float, float, float]]:
    """
    Compute adjusted correlation scores for all candidates.

    Args:
        performance_metric: Performance metric time series
        candidate_metrics: Dictionary of candidate metrics
        max_shift: Maximum time shift
        max_width: Maximum smoothing window
        shift_penalty: Penalty per shift
        smooth_penalty: Penalty per smoothing

    Returns:
        Dictionary of {metric_name: (score, correlation, penalty)}
    """
    results = {
        name: compute_adjusted_correlation(
            performance_metric,
            metric,
            max_shift=max_shift,
            max_width=max_width,
            shift_penalty=shift_penalty,
            smooth_penalty=smooth_penalty
        )
        for name, metric in candidate_metrics.items()
    }

    return results


def filter_by_adjusted_correlation(
    correlation_results: Dict[str, Tuple[float, float, float]],
    min_similarity: float = 0.5,
    graph_refinement_knowledge: Optional[Dict[str, str]] = None
) -> List[str]:
    """
    Filter candidates based on adjusted correlation scores.

    Returns candidates sorted by score in descending order.

    Args:
        correlation_results: Dictionary of correlation results
        min_similarity: Minimum score threshold
        graph_refinement_knowledge: Optional direction constraints (e.g., {'cpu': 'positive'})

    Returns:
        List of candidate metric names sorted by score (descending)
    """
    if graph_refinement_knowledge:
        return _filter_with_knowledge(correlation_results, min_similarity, graph_refinement_knowledge)

    # Simple filtering without knowledge
    candidates_with_scores = [
        (name, score)
        for name, (score, _, _) in correlation_results.items()
        if score >= min_similarity
    ]

    candidates_with_scores.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in candidates_with_scores]


def _filter_with_knowledge(
    correlation_results: Dict[str, Tuple[float, float, float]],
    min_similarity: float,
    knowledge: Dict[str, str]
) -> List[str]:
    """Filter candidates using graph refinement knowledge about correlation direction."""
    candidates_with_scores = []

    for name, (score, corr, _) in correlation_results.items():
        if score < min_similarity:
            continue

        # Find matching metric type in knowledge
        metric_type = next((key for key in knowledge if key in name), None)

        if metric_type:
            expected_sign = knowledge[metric_type]
            # Filter out correlations with wrong sign
            if expected_sign == "negative" and corr > 0:
                continue
            if expected_sign == "positive" and corr < 0:
                continue

        candidates_with_scores.append((name, score))

    candidates_with_scores.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in candidates_with_scores]