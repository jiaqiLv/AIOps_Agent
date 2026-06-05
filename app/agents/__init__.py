"""Agents module - Three-agent architecture with Report Agent

Exports:
- main_graph: The main entry point graph (START → supervisor → END)
- supervisor_agent: Plan-Execute supervisor (planner → executor → reporter)
- detection_agent: Detection subgraph for BLD Metric (ECOD) anomaly detection
- diagnose_agent: Diagnose subgraph for root cause analysis (structured results)
- report_agent: Report agent for NL report generation from structured data
"""

from app.agents.main_graph import main_graph
from app.agents.supervisor_plan_execute import plan_execute_agent as supervisor_agent
from app.agents.detection_agent import detection_agent
from app.agents.diagnose_agent import diagnose_agent
from app.agents.report_agent import report_agent

__all__ = [
    "main_graph",
    "supervisor_agent",
    "detection_agent",
    "diagnose_agent",
    "report_agent",
]
