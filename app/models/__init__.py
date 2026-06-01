"""Models module - unified interface for different LLM providers and state schemas"""

# LLM Models
from app.models.base import BaseLLM, LLMConfig
from app.models.deepseek import DeepSeekModel
from app.models.model_factory import create_model, ModelType

# State Schemas
from app.models.plan_execute_state import PlanExecuteState, PlanStep
from app.models.detection_agent_state import DetectionAgentState
from app.models.react_agent_state import ReactAgentState
from app.models.report_agent_state import ReportAgentState
from app.models.schemas import (
    CsvReaderInput,
    CsvReaderResult,
    RcdInput,
    PcInput,
    AskUserInput,
    AskUserResult,
    RootCauseMetricSchema,
    GraphNodeSchema,
    GraphEdgeSchema,
    PropagationGraphSchema,
    AlgorithmResultSchema,
    DiagnosisResultSchema
)

__all__ = [
    # LLM Models
    "BaseLLM",
    "LLMConfig",
    "DeepSeekModel",
    "create_model",
    "ModelType",
    # State Schemas
    "PlanExecuteState",
    "PlanStep",
    "DetectionAgentState",
    "ReactAgentState",
    "ReportAgentState",
    # Input/Output Schemas
    "CsvReaderInput",
    "CsvReaderResult",
    "RcdInput",
    "PcInput",
    "AskUserInput",
    "AskUserResult",
    "RootCauseMetricSchema",
    "GraphNodeSchema",
    "GraphEdgeSchema",
    "PropagationGraphSchema",
    "AlgorithmResultSchema",
    "DiagnosisResultSchema",
]
