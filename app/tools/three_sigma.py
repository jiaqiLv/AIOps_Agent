"""3-sigma anomaly detection for time-series metrics data.

Uses a baseline window (before inject_time) to compute μ/σ for each metric,
then flags values exceeding μ ± threshold × σ in the detection window (after inject_time).
"""

import json
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np

from app.utils.logger import get_logger
from app.utils.time_utils import format_unix_ts

logger = get_logger(__name__)


def run_three_sigma(
    data: pd.DataFrame,
    inject_time: float,
    baseline_start_minutes: int = 30,
    baseline_end_minutes: int = 60,
    detect_before_minutes: int = 10,
    detect_minutes: int = 10,
    threshold: float = 3.0,
    metric_columns: Optional[List[str]] = None,
    time_column: str = "time",
) -> str:
    """Run 3-sigma anomaly detection on time-series metrics.

    Args:
        data: DataFrame with a Unix-timestamp time column and numeric metric columns.
        inject_time: Fault injection time as Unix timestamp (seconds).
        baseline_start_minutes: Minutes before inject_time where baseline starts (default: 30).
            Baseline window begins at inject_time - baseline_end_minutes*60.
        baseline_end_minutes: Minutes before inject_time where baseline ends (default: 60).
            Baseline window ends at inject_time - baseline_start_minutes*60.
            Default baseline: [inject_time - 60min, inject_time - 30min).
        detect_before_minutes: Minutes before inject_time to include in detection window (default: 10).
        detect_minutes: Minutes after inject_time for the detection window (default: 10).
        threshold: Number of standard deviations (default 3.0).
        metric_columns: Columns to check; if None, all numeric columns except 'time'.
        time_column: Name of the time column.

    Returns:
        JSON string with ranked anomaly list.
    """
    logger.info(
        f"3-sigma: inject_time={inject_time}, "
        f"baseline=[-{baseline_end_minutes}min, -{baseline_start_minutes}min), "
        f"detect=[-{detect_before_minutes}min, +{detect_minutes}min], "
        f"threshold={threshold}"
    )

    try:
        df = data.copy()

        # Ensure time column is numeric (naive Beijing Unix timestamp in seconds)
        if time_column in df.columns and not pd.api.types.is_numeric_dtype(df[time_column]):
            parsed = pd.to_datetime(df[time_column], errors="coerce")
            if parsed.notna().any():
                # Naive Beijing: treat datetime strings as UTC, no timezone offset
                epoch = pd.Timestamp("1970-01-01")
                df[time_column] = (parsed - epoch) / pd.Timedelta(1, "s")
                logger.info(
                    f"Converted time column from datetime strings to naive Beijing timestamps: "
                    f"range [{df[time_column].min():.0f}, {df[time_column].max():.0f}]"
                )
            else:
                # Try direct numeric conversion as fallback
                df[time_column] = pd.to_numeric(df[time_column], errors="coerce")

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

        # Baseline window: [inject_time - baseline_end_offset*60, inject_time - baseline_start_offset*60)
        # Default: [inject_time - 60min, inject_time - 30min) — skip the 30min right before fault
        baseline_start = inject_time - baseline_end_minutes * 60
        baseline_end = inject_time - baseline_start_minutes * 60
        baseline_df = df[
            (df[time_column] >= baseline_start) & (df[time_column] < baseline_end)
        ]

        # Detection window: [inject_time - detect_before_minutes*60, inject_time + detect_minutes*60]
        detect_start = inject_time - detect_before_minutes * 60
        detect_end = inject_time + detect_minutes * 60
        detect_df = df[
            (df[time_column] >= detect_start) & (df[time_column] <= detect_end)
        ]

        if baseline_df.empty:
            return json.dumps({
                "success": False,
                "error": f"Baseline window is empty. inject_time={format_unix_ts(inject_time)}, "
                         f"baseline=[-{baseline_end_minutes}min, -{baseline_start_minutes}min). "
                         f"Data time range: {format_unix_ts(df[time_column].min())} ~ {format_unix_ts(df[time_column].max())}.",
                "anomalies": [],
            }, ensure_ascii=False)

        if detect_df.empty:
            return json.dumps({
                "success": False,
                "error": f"Detection window is empty. inject_time={format_unix_ts(inject_time)}, "
                         f"detect=[-{detect_before_minutes}min, +{detect_minutes}min].",
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
                # Constant baseline — flag any detection value that deviates from mu
                for idx, row in detect_df.iterrows():
                    val = row[col]
                    if pd.isna(val):
                        continue
                    if val != mu:
                        anomalies.append({
                            "metric": col,
                            "index": int(idx),
                            "timestamp": float(row[time_column]),
                            "value": float(val),
                            "baseline_mean": float(mu),
                            "baseline_std": float(sigma),
                            "z_score": float("inf"),
                            "anomaly_type": "sudden_increase" if val > mu else "sudden_decrease",
                        })
                continue

            # Find all points in detection window exceeding threshold
            for idx, row in detect_df.iterrows():
                val = row[col]
                if pd.isna(val):
                    continue
                z = abs(val - mu) / sigma
                if z > threshold:
                    anomalies.append({
                        "metric": col,
                        "index": int(idx),
                        "timestamp": float(row[time_column]),
                        "value": float(val),
                        "baseline_mean": float(mu),
                        "baseline_std": float(sigma),
                        "z_score": float(z),
                        "anomaly_type": "sudden_increase" if val > mu else "sudden_decrease",
                    })

        # Sort by z_score descending (inf values first)
        anomalies.sort(key=lambda x: (x["z_score"] if np.isfinite(x["z_score"]) else float("inf")), reverse=True)

        # Replace inf z_scores with a large sentinel for JSON compatibility
        for a in anomalies:
            if not np.isfinite(a["z_score"]):
                a["z_score"] = 9999.0

        # Group anomalies by metric
        anomalies_by_metric: Dict[str, Dict[str, Any]] = {}
        for a in anomalies:
            metric = a["metric"]
            if metric not in anomalies_by_metric:
                anomalies_by_metric[metric] = {
                    "anomaly_type": a["anomaly_type"],
                    "max_z_score": a["z_score"],
                    "points": [],
                }
            anomalies_by_metric[metric]["points"].append({
                "index": a["index"],
                "timestamp": a["timestamp"],
                "value": a["value"],
                "z_score": a["z_score"],
            })

        logger.info(
            f"3-sigma: {len(anomalies_by_metric)} anomalous metrics, "
            f"{len(anomalies)} anomaly points found out of {len(metric_columns)} checked"
        )

        return json.dumps({
            "success": True,
            "algorithm": "3-sigma",
            "parameters": {
                "inject_time": inject_time,
                "baseline_start_minutes": baseline_start_minutes,
                "baseline_end_minutes": baseline_end_minutes,
                "detect_before_minutes": detect_before_minutes,
                "detect_minutes": detect_minutes,
                "threshold": threshold,
            },
            "baseline_points": len(baseline_df),
            "detection_points": len(detect_df),
            "metrics_checked": len(metric_columns),
            "anomalous_metric_count": len(anomalies_by_metric),
            "anomalies_found": len(anomalies),
            "anomalies": anomalies,
            "anomalies_by_metric": anomalies_by_metric,
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"3-sigma failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "anomalies": [],
        }, ensure_ascii=False)
