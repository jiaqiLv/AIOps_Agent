"""Detection Agent State

State schema for the detection agent that performs anomaly detection
on metrics data (3-Sigma, BLD Metric ECOD).
"""

from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class DetectionAgentState(TypedDict, total=False):
    """State for the detection agent ReAct loop.

    The detection agent loads CSV data and runs anomaly detection
    (3-Sigma / BLD Metric ECOD) to identify abnormal metrics.
    """

    # Core conversation
    messages: Annotated[List[BaseMessage], add_messages]
    task_description: str

    # ReAct control
    iteration_count: int
    max_iterations: int          # default 5
    tool_errors: List[Dict[str, Any]]
    tool_results: Dict[str, Any]

    # Parameters
    csv_file_path: Optional[str]
    csv_headers: Optional[List[str]]
    inject_time: Optional[float]
    abnormal_kpi: Optional[str]

    # Detection result
    three_sigma_result: Optional[Dict[str, Any]]
    bld_metric_result: Optional[Dict[str, Any]]

    # Detection parameters (baseline/detection window, threshold, metrics info)
    detection_parameters: Optional[Dict[str, Any]]

    # Structured anomaly report (per-metric anomaly records)
    anomaly_report: Optional[List[Dict[str, Any]]]

    # Output
    final_response: Optional[str]
