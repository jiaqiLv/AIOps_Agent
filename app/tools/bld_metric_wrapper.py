"""BLD Metric anomaly detection tool.

Wraps the ECOD (Empirical Cumulative Distribution Functions) algorithm
used by the bld_metric system. Uses the first N hours of CSV data as a
training set and detects anomalies in the remaining data.

Algorithm pattern (from bld_metric's WhiteNoiseAnomalyDetection):
  1. Per metric column: extract training window (first train_hours)
  2. Train ECOD on training values
  3. Predict anomaly labels on the full series
  4. Filter out predictions below the training median (omit_lower_anomaly)
"""

import json
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from app.utils.logger import get_logger

logger = get_logger(__name__)


def run_bld_metric_detection(
    data: pd.DataFrame,
    train_hours: float = 1.0,
    contamination: float = 0.001,
    metric_columns: Optional[List[str]] = None,
    time_column: str = "time",
) -> str:
    """Run BLD Metric (ECOD) anomaly detection on time-series metrics.

    Splits each metric column into a training window (first *train_hours*
    hours) and a detection window (everything after). An ECOD model is
    trained per metric and anomalies are flagged in the detection window.

    Args:
        data: DataFrame with a numeric time column and numeric metric columns.
        train_hours: Hours from the start of the data to use for training.
        contamination: Expected fraction of anomalies (ECOD parameter).
        metric_columns: Columns to check; if None, all numeric columns except
            *time_column*.
        time_column: Name of the time column.

    Returns:
        JSON string with per-metric anomaly records.
    """
    logger.info(
        f"BLD_METRIC: train_hours={train_hours}, "
        f"contamination={contamination}"
    )

    try:
        df = data.copy()

        # ── time column ──────────────────────────────────────────
        if time_column not in df.columns:
            return json.dumps({
                "success": False,
                "error": f"Time column '{time_column}' not found. "
                         f"Columns: {df.columns.tolist()}",
                "anomalies": [],
            }, ensure_ascii=False)

        # Ensure time column is numeric (naive Beijing timestamp in seconds)
        if not pd.api.types.is_numeric_dtype(df[time_column]):
            parsed = pd.to_datetime(df[time_column], errors="coerce")
            if parsed.notna().any():
                epoch = pd.Timestamp("1970-01-01")
                df[time_column] = (parsed - epoch) / pd.Timedelta(1, "s")

        df = df.sort_values(time_column).reset_index(drop=True)

        # ── metric columns ───────────────────────────────────────
        if metric_columns is None:
            metric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            metric_columns = [c for c in metric_columns if c != time_column]

        if not metric_columns:
            return json.dumps({
                "success": False,
                "error": "No numeric metric columns found in data.",
                "anomalies": [],
            }, ensure_ascii=False)

        # ── split train / detect ─────────────────────────────────
        time_min = df[time_column].min()
        train_cutoff = time_min + train_hours * 3600

        train_df = df[df[time_column] < train_cutoff]
        detect_df = df[df[time_column] >= train_cutoff]

        if train_df.empty:
            return json.dumps({
                "success": False,
                "error": (
                    f"Training window is empty. "
                    f"train_hours={train_hours}, "
                    f"data time range: {time_min:.0f} ~ {df[time_column].max():.0f}"
                ),
                "anomalies": [],
            }, ensure_ascii=False)

        if detect_df.empty:
            return json.dumps({
                "success": False,
                "error": (
                    f"Detection window is empty after {train_hours}h training. "
                    f"Data only covers {time_min:.0f} ~ {df[time_column].max():.0f}"
                ),
                "anomalies": [],
            }, ensure_ascii=False)

        logger.info(
            f"BLD_METRIC: train={len(train_df)} rows, "
            f"detect={len(detect_df)} rows, "
            f"metrics={len(metric_columns)}"
        )

        # ── lazy import ECOD ─────────────────────────────────────
        try:
            from pyod.models.ecod import ECOD
        except ImportError:
            return json.dumps({
                "success": False,
                "error": "pyod is not installed. Run: pip install pyod",
                "anomalies": [],
            }, ensure_ascii=False)

        anomalies = []
        anomalies_by_metric: Dict[str, Dict[str, Any]] = {}

        for col in metric_columns:
            train_vals = train_df[col].dropna()
            detect_vals = detect_df[col].dropna()

            if train_vals.empty or detect_vals.empty:
                continue

            # Values must be 2D for ECOD
            X_train = train_vals.values.reshape(-1, 1)
            X_detect = detect_vals.values.reshape(-1, 1)

            training_median = float(np.median(X_train))

            try:
                model = ECOD(contamination=contamination)
                model.fit(X_train)

                # Predict on detection window only
                y_pred = model.predict(X_detect)
            except Exception as e:
                logger.warning(f"BLD_METRIC: ECOD failed for '{col}': {e}")
                continue

            # ── omit-lower-anomaly (bld_metric pattern) ─────────
            #  Drop anomaly flags where the observed value is below
            #  the training median (only flag upward spikes).
            anomaly_indices = np.where(y_pred == 1)[0]
            for ai in anomaly_indices:
                val = float(X_detect[ai][0])
                if val <= training_median:
                    y_pred[ai] = 0

            anomaly_count = int((y_pred == 1).sum())
            if anomaly_count == 0:
                continue

            # ── collect anomaly points ───────────────────────────
            detect_indices = detect_vals.index.tolist()
            points: List[Dict[str, Any]] = []
            for ai in np.where(y_pred == 1)[0]:
                idx = detect_indices[ai]
                val = float(X_detect[ai][0])
                score = float(model.decision_function(X_detect[ai:ai + 1])[0])
                points.append({
                    "index": int(idx),
                    "timestamp": float(df.loc[idx, time_column]),
                    "value": val,
                    "score": score,
                })

            anomaly_type = _classify_anomaly_type(points, training_median)

            anomalies_by_metric[col] = {
                "anomaly_type": anomaly_type,
                "max_score": float(max(p["score"] for p in points)),
                "point_count": len(points),
                "training_median": training_median,
                "points": points,
            }
            anomalies.extend(
                {**p, "metric": col, "anomaly_type": anomaly_type}
                for p in points
            )

        # Sort anomalies by score descending
        anomalies.sort(key=lambda x: x["score"], reverse=True)

        logger.info(
            f"BLD_METRIC: {len(anomalies_by_metric)} anomalous metrics, "
            f"{len(anomalies)} anomaly points out of {len(metric_columns)} checked"
        )

        return json.dumps({
            "success": True,
            "algorithm": "bld_metric_ecod",
            "parameters": {
                "train_hours": train_hours,
                "train_cutoff": float(train_cutoff),
                "contamination": contamination,
                "train_rows": len(train_df),
                "detect_rows": len(detect_df),
            },
            "metrics_checked": len(metric_columns),
            "anomalous_metric_count": len(anomalies_by_metric),
            "anomalies_found": len(anomalies),
            "anomalies": anomalies,
            "anomalies_by_metric": anomalies_by_metric,
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"BLD_METRIC failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "anomalies": [],
        }, ensure_ascii=False)


def _classify_anomaly_type(
    points: List[Dict[str, Any]],
    training_median: float,
) -> str:
    """Classify anomaly direction: sudden_increase or sudden_decrease."""
    above = sum(1 for p in points if p["value"] > training_median)
    below = len(points) - above
    return "sudden_increase" if above >= below else "sudden_decrease"
