"""Sub-agent Adapter Registry

Provides a unified interface for invoking sub-agent subgraphs.
Each adapter handles:
1. build_input: Convert plan step input + prior step_results → subgraph input state
2. extract_result: Extract standardized result dict from subgraph output

This replaces the old StructuredTool wrapping in subagent_tools.py.
"""

import json
from typing import Dict, Any

from app.utils.logger import get_logger

logger = get_logger(__name__)


class SubAgentAdapter:
    """Base adapter interface for sub-agents."""

    name: str = ""

    def build_input(self, step_input: dict, step_results: dict) -> dict:
        """Build the subgraph input state from plan step input and prior results."""
        raise NotImplementedError

    def extract_result(self, subgraph_output: dict) -> dict:
        """Extract standardized result dict from subgraph output."""
        raise NotImplementedError


class DetectionAdapter(SubAgentAdapter):
    """Adapter for the Detection Agent (3-Sigma anomaly detection)."""

    name = "detection"

    def build_input(self, step_input: dict, step_results: dict) -> dict:
        return {
            "messages": [],
            "task_description": step_input.get("task_description", ""),
            "csv_file_path": step_input.get("data_path"),
            "inject_time": step_input.get("inject_time"),
            "max_iterations": 5,
            "iteration_count": 0,
            "tool_errors": [],
            "tool_results": {},
        }

    def extract_result(self, output: dict) -> dict:
        # Extract inject_time from three_sigma_result parameters (reliable source)
        three_sigma = output.get("three_sigma_result", {})
        inject_time = output.get("inject_time") or (
            three_sigma.get("parameters", {}).get("inject_time")
        )
        return {
            "success": bool(three_sigma.get("success", False)),
            "summary": output.get("final_response", ""),
            "csv_file_path": output.get("csv_file_path"),
            "inject_time": inject_time,
            "abnormal_kpi": output.get("abnormal_kpi"),
            "three_sigma_result": three_sigma,
            "anomaly_report": output.get("anomaly_report", []),
            "detection_parameters": output.get("detection_parameters"),
        }


class DiagnoseAdapter(SubAgentAdapter):
    """Adapter for the Diagnose Agent (IAF-RCL + KE-FPC root cause analysis)."""

    name = "diagnose"

    def build_input(self, step_input: dict, step_results: dict) -> dict:
        # Check if a prior detection step result should be passed
        detection_data = {}
        from_step = step_input.get("from_step")
        if from_step and from_step in step_results:
            detection_data = step_results[from_step]

        # Build enriched task_description with detection context
        task = step_input.get("task_description", "")
        anomaly_report = detection_data.get("anomaly_report") if detection_data else None
        if anomaly_report:
            anomaly_metrics = ", ".join(
                r.get("metric", "?") for r in anomaly_report[:5]
            )
            task = f"{task}\n\n## 异常检测结果\n检出入取异常指标: {anomaly_metrics}"

        return {
            "messages": [],
            "task_description": task,
            "csv_file_path": (
                step_input.get("data_path")
                or (detection_data.get("csv_file_path") if detection_data else None)
            ),
            "inject_time": (
                step_input.get("inject_time")
                or (detection_data.get("inject_time") if detection_data else None)
            ),
            "abnormal_kpi": (
                step_input.get("abnormal_kpi")
                or (detection_data.get("abnormal_kpi") if detection_data else None)
            ),
            "max_iterations": 10,
            "iteration_count": 0,
            "tool_errors": [],
            "tool_results": {},
        }

    def extract_result(self, output: dict) -> dict:
        integrated = {}
        raw = output.get("integrated_result")
        if raw:
            try:
                integrated = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                pass

        return {
            "success": bool(
                output.get("rcd_result") or output.get("pc_result")
                or integrated.get("rcd_result") or integrated.get("pc_result")
            ),
            "rcd_result": integrated.get("rcd_result") or output.get("rcd_result"),
            "pc_result": integrated.get("pc_result") or output.get("pc_result"),
            "graph_visualizations": (
                integrated.get("graph_visualizations")
                or output.get("graph_visualizations", [])
            ),
            "summary": output.get("final_response", ""),
            "csv_file_path": integrated.get("csv_file_path") or output.get("csv_file_path"),
            "inject_time": integrated.get("inject_time") or output.get("inject_time"),
            "abnormal_kpi": integrated.get("abnormal_kpi") or output.get("abnormal_kpi"),
            "tool_errors": integrated.get("tool_errors") or output.get("tool_errors", []),
            # New structured fields
            "fault_type": integrated.get("fault_type") or output.get("fault_type"),
            "root_causes": integrated.get("root_causes") or output.get("root_causes", []),
            "propagation_path": integrated.get("propagation_path") or output.get("propagation_path", []),
        }


class ReportAdapter(SubAgentAdapter):
    """Adapter for the Report Agent (NL report from structured data).

    Not invoked as a separate plan step — the supervisor reporter uses
    this adapter to build report agent input from all step_results.
    """

    name = "report"

    def build_input(self, step_input: dict, step_results: dict) -> dict:
        # Collect structured data from all step_results
        detection_data = {}
        diagnose_data = {}
        task_description = step_input.get("task_description", "")
        csv_file_path = None
        inject_time = None
        abnormal_kpi = None
        graph_visualizations = []
        tool_errors = []

        for step_id, result in step_results.items():
            if result.get("anomaly_report"):
                detection_data = result
                csv_file_path = csv_file_path or result.get("csv_file_path")
                inject_time = inject_time or result.get("inject_time")
                abnormal_kpi = abnormal_kpi or result.get("abnormal_kpi")
            if result.get("root_causes"):
                diagnose_data = result
                csv_file_path = csv_file_path or result.get("csv_file_path")
                inject_time = inject_time or result.get("inject_time")
                abnormal_kpi = abnormal_kpi or result.get("abnormal_kpi")
                graph_visualizations = result.get("graph_visualizations", [])
            tool_errors.extend(result.get("tool_errors", []))

        return {
            "messages": [],
            "task_description": task_description,
            "detection_anomaly_report": detection_data.get("anomaly_report", []),
            "detection_parameters": detection_data.get("detection_parameters"),
            "diagnose_fault_type": diagnose_data.get("fault_type"),
            "diagnose_root_causes": diagnose_data.get("root_causes", []),
            "diagnose_propagation_path": diagnose_data.get("propagation_path", []),
            "graph_visualizations": graph_visualizations,
            "csv_file_path": csv_file_path,
            "inject_time": inject_time,
            "abnormal_kpi": abnormal_kpi,
            "tool_errors": tool_errors,
        }

    def extract_result(self, output: dict) -> dict:
        return {
            "success": bool(output.get("final_response")),
            "summary": output.get("final_response", ""),
        }


# ==================== Registry ====================

REGISTRY: Dict[str, SubAgentAdapter] = {
    "detection": DetectionAdapter(),
    "diagnose": DiagnoseAdapter(),
    "report": ReportAdapter(),
}


def get_adapter(agent_name: str) -> SubAgentAdapter:
    """Get adapter by agent name. Raises ValueError if not found."""
    if agent_name not in REGISTRY:
        raise ValueError(f"Unknown sub-agent: {agent_name}")
    return REGISTRY[agent_name]


def get_subgraph(agent_name: str):
    """Get the compiled subgraph for a given agent name.

    Imports are deferred to avoid circular dependencies.
    Returns a LazyGraph proxy that builds on first use.
    """
    if agent_name == "detection":
        from app.agents.detection_agent import detection_agent
        return detection_agent
    elif agent_name == "diagnose":
        from app.agents.diagnose_agent import diagnose_agent
        return diagnose_agent
    elif agent_name == "report":
        from app.agents.report_agent import report_agent
        return report_agent
    else:
        raise ValueError(f"Unknown sub-agent: {agent_name}")
