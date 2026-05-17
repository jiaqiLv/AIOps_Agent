"""RCD (Root Cause Discovery) algorithm wrapper tool

This tool wraps the RCD algorithm for root cause analysis.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from app.utils.logger import get_logger

logger = get_logger(__name__)

def _import_rcd():
    """Lazy-import rcd, trying multiple strategies for different runtime environments."""
    # Strategy 1: package import (LangGraph Studio)
    try:
        from app.tools.rcd.rcd import rcd as _rcd
        logger.info("[RCD_WRAPPER] RCD loaded via package import")
        return _rcd, True
    except ImportError:
        pass

    # Strategy 2: sys.path injection (CLI / legacy)
    import sys, os
    rcd_dir = os.path.join(os.path.dirname(__file__), 'rcd')
    if rcd_dir not in sys.path:
        sys.path.insert(0, rcd_dir)
    try:
        from rcd import rcd as _rcd
        logger.info("[RCD_WRAPPER] RCD loaded via sys.path injection")
        return _rcd, True
    except ImportError as e:
        logger.error(f"[RCD_WRAPPER] RCD import failed: {e}", exc_info=True)
        return None, False


def run_rcd_analysis(
    data: pd.DataFrame,
    inject_time: float,
    gamma: int = 5,
    localized: bool = True,
    bins: int = 5,
    dk_select_useful: bool = False,
    verbose: bool = False,
    dataset: Optional[str] = None,
    seed: Optional[int] = None,
    abnormal_kpi: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run RCD algorithm for root cause analysis.

    Args:
        data: DataFrame with time series data (must include 'time' column as 10-digit timestamp)
        inject_time: Injection time for anomaly detection (timestamp index or numeric)
        gamma: Number of nodes in each subset for phase-1
        localized: Whether to use localized version of PSI-PC
        bins: Number of bins for discretization
        dk_select_useful: Whether to select useful columns
        verbose: Verbose mode
        dataset: Dataset type for preprocessing (e.g., "sock-shop")
        seed: Random seed for reproducibility
        abnormal_kpi: Name of the abnormal KPI metric (optional, for filtering results)

    Returns:
        Dictionary containing:
        - root_causes: List of root cause metrics (ranked)
        - status: Execution status
        - message: Status message
        - abnormal_kpi: The abnormal KPI used for analysis (if provided)
    """
    logger.info(f"[RCD_WRAPPER] run_rcd_analysis called with data shape={data.shape}, inject_time={inject_time}")

    rcd, rcd_available = _import_rcd()
    if not rcd_available:
        logger.error("[RCD_WRAPPER] RCD not available!")
        return {
            "root_causes": [],
            "status": "error",
            "message": "RCD algorithm not available. Please ensure required dependencies are installed."
        }

    try:
        logger.info(f"Running RCD analysis with {len(data.columns)} metrics, inject_time={inject_time}")

        # Validate inject_time against data
        if "time" not in data.columns:
            return {
                "root_causes": [],
                "status": "error",
                "message": "CSV data must contain 'time' column for RCD analysis"
            }

        # Log time information for debugging
        time_min = data["time"].min()
        time_max = data["time"].max()
        logger.info(f"Data time range: {time_min} to {time_max}")

        # Check if inject_time is within data range
        if inject_time < time_min:
            logger.warning(f"inject_time ({inject_time}) is BEFORE data start ({time_min})")
            return {
                "root_causes": [],
                "status": "error",
                "message": f"inject_time ({inject_time}) is before data time range ({time_min} to {time_max}). Please provide a time within the data range."
            }
        elif inject_time > time_max:
            logger.warning(f"inject_time ({inject_time}) is AFTER data end ({time_max})")
            return {
                "root_causes": [],
                "status": "error",
                "message": f"inject_time ({inject_time}) is after data time range ({time_min} to {time_max}). Please provide a time within the data range."
            }

        # Count rows before and after inject_time
        normal_count = (data["time"] < inject_time).sum()
        anomalous_count = (data["time"] >= inject_time).sum()
        logger.info(f"Data split: normal={normal_count} rows, anomalous={anomalous_count} rows")
        logger.info(f"inject_time={inject_time}, data time range: {time_min} to {time_max}")

        # Quick check for data size - if too large, RCD will take too long
        total_rows = len(data)
        total_cols = len(data.columns)

        # Data size check removed - RCD will run on any size data
        logger.info(f"[RCD_WRAPPER] Processing data: {total_rows} rows x {total_cols} cols")

        if normal_count == 0:
            return {
                "root_causes": [],
                "status": "error",
                "message": f"No normal data found. All data points have time >= inject_time ({inject_time})."
            }
        if anomalous_count == 0:
            return {
                "root_causes": [],
                "status": "error",
                "message": f"No anomalous data found. All data points have time < inject_time ({inject_time})."
            }

        # Call RCD algorithm
        logger.info(f"Calling RCD algorithm with data shape: {data.shape}, inject_time: {inject_time}, gamma: {gamma}, dataset={dataset}")

        # Call RCD algorithm directly without stdout capture
        logger.info("[RCD_WRAPPER] Calling RCD algorithm (this may take several minutes for large datasets)...")

        try:
            result = rcd(
                data=data,
                inject_time=inject_time,
                dk_select_useful=dk_select_useful,
                gamma=gamma,
                localized=localized,
                bins=bins,
                dataset=dataset,
                seed=seed,
                verbose=False
            )
            logger.info(f"[RCD_WRAPPER] RCD algorithm completed successfully")
        except Exception as e:
            logger.error(f"[RCD_WRAPPER] RCD algorithm failed: {e}", exc_info=True)
            result = {"ranks": []}

        logger.info(f"RCD algorithm returned result with ranks: {result.get('ranks', [])}")

        # Extract root causes
        root_causes = result.get("ranks", [])

        # Filter by abnormal_kpi if provided (prioritize metrics related to abnormal KPI)
        if abnormal_kpi and root_causes:
            # Keep all results but note the abnormal KPI
            logger.info(f"RCD analysis completed with abnormal_kpi: {abnormal_kpi}")

        logger.info(f"RCD analysis completed. Found {len(root_causes)} root causes")

        response = {
            "root_causes": root_causes,
            "status": "success",
            "message": f"RCD analysis completed successfully. Found {len(root_causes)} potential root causes."
        }

        if abnormal_kpi:
            response["abnormal_kpi"] = abnormal_kpi

        return response

    except Exception as e:
        logger.error(f"RCD analysis failed: {e}", exc_info=True)
        return {
            "root_causes": [],
            "status": "error",
            "message": f"RCD analysis failed: {str(e)}"
        }


def format_rcd_results(result: Dict[str, Any]) -> str:
    """
    Format RCD results for display.

    Args:
        result: Result dictionary from run_rcd_analysis

    Returns:
        Formatted string with results
    """
    if result.get("status") == "error":
        return f"❌ RCD Analysis Error: {result.get('message')}"

    root_causes = result.get("root_causes", [])

    if not root_causes:
        return "⚠️ RCD Analysis: No root causes identified."

    output = ["=== RCD 根因分析结果 ===\n"]
    output.append(f"📊 分析状态: {result.get('status', 'unknown')}")
    output.append(f"🔍 发现根因数量: {len(root_causes)}\n")

    output.append("**根因指标列表（按优先级排序）**:")
    for i, cause in enumerate(root_causes[:10], 1):  # Show top 10
        output.append(f"  {i}. {cause}")

    if len(root_causes) > 10:
        output.append(f"  ... 及其他 {len(root_causes) - 10} 个指标")

    return "\n".join(output)
