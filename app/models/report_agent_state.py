"""Report Agent State

State schema for the report agent that generates natural-language
reports from structured detection and diagnose outputs.
"""

from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class ReportAgentState(TypedDict, total=False):
    """State for the report agent.

    Takes structured data from detection and diagnose agents
    and generates a unified natural-language root cause analysis report.
    """

    # Core conversation
    messages: Annotated[List[BaseMessage], add_messages]

    # Input
    task_description: str

    # Structured data from detection agent
    detection_anomaly_report: List[Dict[str, Any]]
    """Per-metric anomaly records from detection agent"""

    detection_parameters: Optional[Dict[str, Any]]
    """Detection algorithm parameters (baseline window, detection window, threshold, metrics counts)"""

    # Structured data from diagnose agent
    diagnose_fault_type: Optional[str]
    """Inferred fault type from diagnose agent"""

    diagnose_root_causes: List[Dict[str, Any]]
    """Root cause records with metric, confidence, algorithms, reason"""

    diagnose_propagation_path: List[Dict[str, Any]]
    """Causal propagation edges"""

    # Shared metadata
    graph_visualizations: List[Dict[str, Any]]
    csv_file_path: Optional[str]
    inject_time: Optional[float]
    abnormal_kpi: Optional[str]

    # Execution context
    tool_errors: List[Dict[str, Any]]

    # Output
    final_response: Optional[str]
