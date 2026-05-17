"""Common schemas for the AIOps Agent system"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ==================== Input Schemas ====================

class CsvReaderInput(BaseModel):
    """Input schema for CSV reader tool"""
    data_path: str = Field(..., description="Path to the CSV data file")


class CsvReaderResult(BaseModel):
    """Result schema for CSV reader tool"""
    success: bool
    columns: List[str] = Field(default_factory=list)
    shape: Optional[tuple] = None
    preview: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


class RcdInput(BaseModel):
    """Input schema for rcd_tool (IAF-RCL algorithm)"""
    data_path: str = Field(..., description="Path to CSV data file")
    fault_injection_time: str = Field(..., description="Fault injection time")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)


class PcInput(BaseModel):
    """Input schema for pc_tool (KE-FPC algorithm)"""
    data_path: str = Field(..., description="Path to CSV data file")
    abnormal_kpi: str = Field(..., description="Abnormal KPI metric name")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)


class DiagnoseSubagentInput(BaseModel):
    """Input schema for diagnose subagent tool"""
    data_path: Optional[str] = Field(None, description="Direct path to CSV file")
    benchmark: Optional[str] = Field(None, description="Dataset name")
    instance: Optional[str] = Field(None, description="Instance name")
    case: Optional[str] = Field(None, description="Case identifier")
    fault_injection_time: str = Field(..., description="Fault injection time")
    abnormal_kpi: str = Field(..., description="Abnormal KPI metric name")


class AskUserInput(BaseModel):
    """Input schema for ask_user tool"""
    question: str = Field(..., description="Question to ask the user")
    partial_state: Optional[Dict[str, Any]] = Field(default_factory=dict)


class AskUserResult(BaseModel):
    """Result schema for ask_user tool"""
    status: str = "interrupted"
    question: str
    partial_state: Dict[str, Any] = Field(default_factory=dict)


# ==================== Output Schemas ====================

class RootCauseMetricSchema(BaseModel):
    """Schema for root cause metric"""
    metric: str
    rank: int
    score: float
    supported_by: List[str]
    reason: str


class GraphNodeSchema(BaseModel):
    """Schema for graph node"""
    id: str
    type: str


class GraphEdgeSchema(BaseModel):
    """Schema for graph edge"""
    source: str
    target: str
    score: float
    reason: Optional[str] = None


class PropagationGraphSchema(BaseModel):
    """Schema for propagation graph"""
    nodes: List[GraphNodeSchema]
    edges: List[GraphEdgeSchema]
    paths: List[Dict[str, Any]] = Field(default_factory=list)


class AlgorithmResultSchema(BaseModel):
    """Schema for algorithm result"""
    success: bool
    algorithm: str
    root_cause_metrics: List[RootCauseMetricSchema] = Field(default_factory=list)
    propagation_graph: Optional[PropagationGraphSchema] = None
    raw_output: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class DiagnosisResultSchema(BaseModel):
    """Schema for final diagnosis result"""
    summary: str
    fault_type: str
    root_cause_metrics: List[RootCauseMetricSchema] = Field(default_factory=list)
    propagation_graph: Optional[PropagationGraphSchema] = None
    fault_description: str
    filtered_items: Dict[str, Any] = Field(default_factory=dict)
    confidence: float
