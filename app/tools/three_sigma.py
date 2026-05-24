"""3-sigma anomaly detection for time-series metrics data.

Uses a baseline window (before inject_time) to compute μ/σ for each metric,
then flags values exceeding μ ± threshold × σ in the detection window (after inject_time).
"""

import json
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np

from app.utils.logger import get_logger

logger = get_logger(__name__)


def run_three_sigma(
    data: pd.DataFrame,
    inject_time: float,
    baseline_minutes: int = 5,
    detect_minutes: int = 5,
    threshold: float = 3.0,
    metric_columns: Optional[List[str]] = None,
    time_column: str = "time",
) -> str:
    """Run 3-sigma anomaly detection on time-series metrics.

    Args:
        data: DataFrame with a Unix-timestamp time column and numeric metric columns.
        inject_time: Fault injection time as Unix timestamp (seconds).
            Splits baseline (< inject_time) from detection window (>= inject_time).
        baseline_minutes: Minutes before inject_time for the baseline window.
        detect_minutes: Minutes after inject_time for the detection window.
        threshold: Number of standard deviations (default 3.0).
        metric_columns: Columns to check; if None, all numeric columns except 'time'.
        time_column: Name of the time column.

    Returns:
        JSON string with ranked anomaly list.
    """
    logger.info(
        f"3-sigma: inject_time={inject_time}, "
        f"baseline={baseline_minutes}min, detect={detect_minutes}min, threshold={threshold}"
    )

    try:
        df = data.copy()

        if time_column not in df.columns:
            return json.dumps({
                "success": False,
                "error": f"Time column '{time_column}' not found in data. Columns: {df.columns.tolist()}",
                "anomalies": [],
            }, ensure_ascii=False)

        # Determine metric columns
        if metric_columns is None:
            metric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            metric_columns = [c for c in metric_columns if c != time_column]

        if not metric_columns:
            return json.dumps({
                "success": False,
                "error": "No numeric metric columns found in data.",
                "anomalies": [],
            }, ensure_ascii=False)

        # Baseline window: [inject_time - baseline_minutes*60, inject_time)
        baseline_start = inject_time - baseline_minutes * 60
        baseline_df = df[
            (df[time_column] >= baseline_start) & (df[time_column] < inject_time)
        ]

        # Detection window: [inject_time, inject_time + detect_minutes*60]
        detect_end = inject_time + detect_minutes * 60
        detect_df = df[
            (df[time_column] >= inject_time) & (df[time_column] <= detect_end)
        ]

        if baseline_df.empty:
            return json.dumps({
                "success": False,
                "error": f"Baseline window is empty. Check inject_time ({inject_time}) "
                         f"and baseline_minutes ({baseline_minutes}). "
                         f"Data time range: {df[time_column].min()} - {df[time_column].max()}.",
                "anomalies": [],
            }, ensure_ascii=False)

        if detect_df.empty:
            return json.dumps({
                "success": False,
                "error": f"Detection window is empty. Check inject_time ({inject_time}) "
                         f"and detect_minutes ({detect_minutes}).",
                "anomalies": [],
            }, ensure_ascii=False)

        anomalies = []

        for col in metric_columns:
            # Drop NaNs for safe computation
            baseline_vals = baseline_df[col].dropna()
            detect_vals = detect_df[col].dropna()

            if baseline_vals.empty or detect_vals.empty:
                continue

            mu = baseline_vals.mean()
            sigma = baseline_vals.std(ddof=1)  # sample std

            if sigma == 0 or np.isnan(sigma):
                continue  # constant metric, no variance

            # Find all points in detection window exceeding threshold
            for idx, row in detect_df.iterrows():
                val = row[col]
                if pd.isna(val):
                    continue
                z = abs(val - mu) / sigma
                if z > threshold:
                    anomalies.append({
                        "metric": col,
                        "timestamp": float(row[time_column]),
                        "value": float(val),
                        "baseline_mean": float(mu),
                        "baseline_std": float(sigma),
                        "z_score": float(z),
                    })

        # Sort by max z_score per metric, then by timestamp
        anomalies.sort(key=lambda x: x["z_score"], reverse=True)

        # Deduplicate: keep only the max z_score entry per metric
        seen = set()
        deduped = []
        for a in anomalies:
            if a["metric"] not in seen:
                deduped.append(a)
                seen.add(a["metric"])

        logger.info(f"3-sigma: {len(deduped)} anomalous metrics found out of {len(metric_columns)} checked")

        return json.dumps({
            "success": True,
            "algorithm": "3-sigma",
            "parameters": {
                "inject_time": inject_time,
                "baseline_minutes": baseline_minutes,
                "detect_minutes": detect_minutes,
                "threshold": threshold,
            },
            "baseline_points": len(baseline_df),
            "detection_points": len(detect_df),
            "metrics_checked": len(metric_columns),
            "anomalies_found": len(deduped),
            "anomalies": deduped,
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"3-sigma failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "anomalies": [],
        }, ensure_ascii=False)
