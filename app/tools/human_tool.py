"""Human-in-the-loop tool for requesting user input"""

from typing import Dict, Any, Optional, List
from app.utils.logger import get_logger

logger = get_logger(__name__)


def ask_user(
    question: str,
    partial_state: Optional[Dict[str, Any]] = None,
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Request user input for missing parameters.

    This function creates an interrupt that requests the user
    to provide missing information.

    Args:
        question: The question to ask the user
        partial_state: Current partial state information
        context: Additional context for the user

    Returns:
        Dictionary containing:
        - status: "interrupted"
        - question: str - The question asked
        - partial_state: dict - Current partial state
        - context: str - Additional context
    """
    logger.info(f"Requesting user input: {question[:100]}...")

    return {
        "status": "interrupted",
        "question": question,
        "partial_state": partial_state or {},
        "context": context,
        "message": f"Please provide: {question}"
    }


def generate_missing_params_question(missing_fields: List[str], partial_state: Dict[str, Any]) -> str:
    """
    Generate a user-friendly question for missing parameters.

    Args:
        missing_fields: List of missing field names
        partial_state: Current partial state

    Returns:
        User-friendly question string
    """
    field_descriptions = {
        "data_path": "CSV data file path (e.g., data/RE1-OB/adservice_cpu/case_001/data.csv)",
        "fault_injection_time": "Fault injection time (e.g., 100 or 2024-01-01T10:30:00)",
        "abnormal_kpi": "Abnormal KPI metric name (e.g., frontend_error_rate)",
        "benchmark": "Dataset name (e.g., RE1-OB)",
        "instance": "Instance name (e.g., adservice_cpu)",
        "case": "Case identifier (e.g., case_001)"
    }

    question_parts = []

    # Check what information we already have
    has_path_info = any(k in partial_state for k in ["data_path", "benchmark", "instance", "case"])
    has_time = "fault_injection_time" in partial_state or "inject_time" in partial_state
    has_kpi = "abnormal_kpi" in partial_state

    if not has_path_info:
        question_parts.append("Please provide the data file location:")
        question_parts.append("  - Either: data_path (full path to CSV file)")
        question_parts.append("  - Or: benchmark + instance + case")

    if "fault_injection_time" in missing_fields:
        question_parts.append("Please provide the fault_injection_time (when the fault occurred)")

    if "abnormal_kpi" in missing_fields:
        question_parts.append("Please provide the abnormal_kpi (which metric is abnormal)")

    return "\n".join(question_parts) if question_parts else "Please provide the missing information."


def format_user_feedback(user_feedback: str, partial_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format user feedback and update partial state.

    Args:
        user_feedback: User's response text
        partial_state: Current partial state to update

    Returns:
        Updated state dictionary
    """
    import re

    updated_state = partial_state.copy()

    # Try to extract data_path
    if not updated_state.get("data_path"):
        path_match = re.search(r'([./].*?\.csv)', user_feedback, re.IGNORECASE)
        if path_match:
            updated_state["data_path"] = path_match.group(1)

    # Try to extract fault_injection_time
    if not updated_state.get("fault_injection_time"):
        # Support both numeric timestamp and datetime string like '2025-11-16 11:10:00'
        time_match = re.search(
            r'(?:fault_time|inject_time|注入时间)[：:]\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})',
            user_feedback, re.IGNORECASE
        )
        if time_match:
            updated_state["fault_injection_time"] = time_match.group(1)
        else:
            time_match = re.search(r'(?:fault_time|inject_time|注入时间)[：:]\s*(\d+\.?\d*)', user_feedback, re.IGNORECASE)
            if time_match:
                updated_state["fault_injection_time"] = time_match.group(1)

    # Try to extract abnormal_kpi
    if not updated_state.get("abnormal_kpi"):
        kpi_match = re.search(r'(?:abnormal_kpi|异常KPI|异常指标)[：:]\s*(\w+)', user_feedback, re.IGNORECASE)
        if kpi_match:
            updated_state["abnormal_kpi"] = kpi_match.group(1)

    return updated_state
