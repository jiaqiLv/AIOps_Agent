"""LangChain Tool Adapters

This module provides utilities to convert registered tools into LangChain
StructuredTool instances for use with LangChain's ToolNode and bind_tools().

The adapters handle:
- Loading tool functions from the ToolRegistry
- Creating Pydantic schemas from tool configuration
- Wrapping tool functions for JSON serialization
- Managing CSV data caching for IAF-RCL/KE-FPC tools
"""

import json
import time
import pandas as pd
from typing import Dict, Any, List, Optional, Callable, Type
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from typing import Union
from app.tools.tool_registry import ToolRegistry
from app.utils.logger import get_logger
from app.utils.path_resolver import resolve_data_path
from app.utils.time_utils import format_unix_ts

logger = get_logger(__name__)

# Module-level agent name context for tool tracing
_current_agent_name: str = "react_agent"


def set_agent_name(name: str) -> None:
    """Set the current agent name for tool call tracing attribution."""
    global _current_agent_name
    _current_agent_name = name


def get_agent_name() -> str:
    """Get the current agent name."""
    return _current_agent_name


# ==================== Time Conversion ====================

def _convert_inject_time(inject_time: Union[float, int, str]) -> float:
    """Convert inject_time to naive Beijing Unix timestamp.

    Naive Beijing timestamps treat Beijing time as UTC (no +8h offset).
    "2026-01-05 05:48:00" → seconds from epoch to 2026-01-05 05:48:00 (as UTC).

    Accepts:
    - float/int: used directly (assumed already in the correct convention)
    - str: human-readable datetime string (Beijing time), parsed as naive UTC

    Returns:
        Naive Beijing Unix timestamp as float
    """
    if isinstance(inject_time, (int, float)):
        return float(inject_time)

    if isinstance(inject_time, str):
        try:
            ts = pd.to_datetime(inject_time)
            # Naive: treat Beijing time as UTC — no timezone offset
            epoch = pd.Timestamp("1970-01-01")
            return float((ts - epoch) / pd.Timedelta(1, "s"))
        except Exception as e:
            raise ValueError(
                f"无法解析 inject_time '{inject_time}': {e}。"
                f"请使用格式如 '2026-01-05 05:48:00'"
            )

    return float(inject_time)


# ==================== CSV Data Cache ====================

_csv_data_cache: Dict[str, pd.DataFrame] = {}


def get_cached_csv(data_path: str) -> Optional[pd.DataFrame]:
    """Get cached CSV data for a path."""
    return _csv_data_cache.get(data_path)


def cache_csv_data(data_path: str, df: pd.DataFrame) -> None:
    """Cache CSV data for a path."""
    _csv_data_cache[data_path] = df


def clear_csv_cache() -> None:
    """Clear all cached CSV data."""
    global _csv_data_cache
    _csv_data_cache = {}


# ==================== Tool Wrappers ====================

def _wrap_csv_reader(func: Callable) -> Callable:
    """Wrap CSV reader tool to cache results."""
    def wrapper(data_path: str, **kwargs) -> str:
        logger.info(f"TOOL: csv_reader_tool called with data_path={data_path}")

        try:
            resolved_path = resolve_data_path(data_path=data_path)
            if not resolved_path:
                resolved_path = data_path

            df = pd.read_csv(resolved_path)

            # Remove duplicate columns
            dup_cols = df.columns[df.columns.duplicated()].unique().tolist()
            if dup_cols:
                logger.warning(f"Removing {len(dup_cols)} duplicate column(s)")
                df = df.loc[:, ~df.columns.duplicated()]

            # Normalize time column — naive Beijing timestamps (treat Beijing time as UTC)
            if 'time' in df.columns:
                if not pd.api.types.is_numeric_dtype(df['time']):
                    try:
                        time_series = pd.to_datetime(df['time'])
                        # Naive: treat Beijing time as UTC, no timezone localization
                        epoch = pd.Timestamp("1970-01-01")
                        df['time'] = (time_series - epoch) / pd.Timedelta(1, "s")
                        logger.info(
                            f"Time column converted to naive Beijing timestamp: "
                            f"{format_unix_ts(df['time'].min())} ~ {format_unix_ts(df['time'].max())}"
                        )
                    except Exception as e:
                        logger.warning(f"Could not convert time column: {e}")

            # Cache the data
            cache_csv_data(resolved_path, df)

            result = {
                "success": True,
                "data_path": resolved_path,
                "shape": [df.shape[0], df.shape[1]],
                "columns": df.columns.tolist(),
                "metric_columns": df.select_dtypes(include=['number']).columns.tolist(),
                "time_range": f"{format_unix_ts(df['time'].min())} ~ {format_unix_ts(df['time'].max())}" if "time" in df.columns else "N/A",
            }
            return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            logger.error(f"CSV reader failed: {e}")
            return json.dumps({"success": False, "error": str(e), "data_path": data_path}, ensure_ascii=False)

    return wrapper


def _wrap_three_sigma(func: Callable) -> Callable:
    """Wrap three_sigma tool to use cached CSV data."""
    def wrapper(inject_time: Union[float, str],
                baseline_start_minutes: int = 30, baseline_end_minutes: int = 60,
                detect_before_minutes: int = 10, detect_minutes: int = 10,
                threshold: float = 3.0, metric_columns: Optional[List[str]] = None, **kwargs) -> str:
        logger.info(f"TOOL: three_sigma_tool called with inject_time={inject_time} (type={type(inject_time).__name__})")

        if not _csv_data_cache:
            return json.dumps({
                "success": False,
                "error": "CSV data not loaded. Please call csv_reader_tool first.",
                "anomalies": [],
            }, ensure_ascii=False)

        df = list(_csv_data_cache.values())[0]

        try:
            inject_time_ts = _convert_inject_time(inject_time)
            result = func(
                data=df,
                inject_time=inject_time_ts,
                baseline_start_minutes=baseline_start_minutes,
                baseline_end_minutes=baseline_end_minutes,
                detect_before_minutes=detect_before_minutes,
                detect_minutes=detect_minutes,
                threshold=threshold,
                metric_columns=metric_columns,
            )
            if isinstance(result, dict):
                return json.dumps(result, ensure_ascii=False)
            return result

        except Exception as e:
            logger.error(f"3-sigma tool failed: {e}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "anomalies": [],
            }, ensure_ascii=False)

    return wrapper


def _wrap_rcd_tool(func: Callable) -> Callable:
    """Wrap RCD tool to use cached CSV data."""
    def wrapper(inject_time: Union[float, str], gamma: int = 5, abnormal_kpi: Optional[str] = None, **kwargs) -> str:
        logger.info(f"TOOL: rcd_tool called with inject_time={inject_time} (type={type(inject_time).__name__})")

        # Get cached CSV data
        if not _csv_data_cache:
            return json.dumps({
                "success": False,
                "error": "CSV data not loaded. Please call csv_reader_tool first.",
                "root_causes": []
            }, ensure_ascii=False)

        df = list(_csv_data_cache.values())[0]  # Get first (and should be only) cached dataset

        try:
            inject_time_ts = _convert_inject_time(inject_time)
            result = func(
                data=df,
                inject_time=inject_time_ts,
                gamma=gamma,
                localized=True,
                bins=5,
                abnormal_kpi=abnormal_kpi,
                verbose=False
            )
            result["success"] = result.get("status") == "success"
            return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            logger.error(f"RCD tool failed: {e}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "root_causes": []
            }, ensure_ascii=False)

    return wrapper


def _wrap_pc_tool(func: Callable) -> Callable:
    """Wrap PC tool to use cached CSV data."""
    def wrapper(alpha: float = 0.05, abnormal_kpi: Optional[str] = None, **kwargs) -> str:
        logger.info(f"TOOL: pc_tool called with alpha={alpha}, abnormal_kpi={abnormal_kpi}")

        # Get cached CSV data
        if not _csv_data_cache:
            return json.dumps({
                "success": False,
                "error": "CSV data not loaded. Please call csv_reader_tool first.",
                "root_causes": [],
                "edges": []
            }, ensure_ascii=False)

        df = list(_csv_data_cache.values())[0]  # Get first (and should be only) cached dataset

        try:
            result = func(
                data=df,
                alpha=alpha,
                abnormal_kpi=abnormal_kpi,
                verbose=False
            )
            result["success"] = result.get("status") == "success"
            return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            logger.error(f"PC tool failed: {e}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "root_causes": [],
                "edges": []
            }, ensure_ascii=False)

    return wrapper


def _wrap_graph_visualization(func: Callable) -> Callable:
    """Wrap graph visualization tool to return JSON string.

    Strips large html_content to keep ToolMessage compact while
    preserving png_base64 for inline Studio display.
    """
    def wrapper(**kwargs) -> str:
        try:
            result = func(**kwargs)
            if isinstance(result, dict):
                # Strip large fields not needed in ToolMessage;
                # png_base64 is kept for inline Studio display.
                _strip = {"html_content", "base64_content"}
                light = {k: v for k, v in result.items()
                         if k not in _strip}
                return json.dumps(light, ensure_ascii=False, default=str)
            return json.dumps({"success": False, "error": "unexpected return type"},
                              ensure_ascii=False)
        except Exception as e:
            logger.error(f"Graph visualization tool failed: {e}")
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)
    return wrapper


# ==================== Tool Trace Wrapper ====================

def _trace_tool_wrapper(tool_name: str, func: Callable) -> Callable:
    """Wrap a tool function to trace its execution via TraceLogger.

    Records start time, logs args, calls the original function,
    logs result, and records duration.
    """
    from app.utils.llm_logger import get_trace_logger

    def wrapper(**kwargs):
        agent = get_agent_name()
        start = time.monotonic()
        error = None
        result = None
        try:
            result = func(**kwargs)
            return result
        except Exception as e:
            error = str(e)
            raise
        finally:
            duration_ms = (time.monotonic() - start) * 1000
            try:
                trace = get_trace_logger()
                # Serialize args — convert non-serializable to str
                safe_args = {}
                for k, v in kwargs.items():
                    try:
                        json.dumps(v, default=str)
                        safe_args[k] = v
                    except (TypeError, ValueError):
                        safe_args[k] = str(v)
                trace.log_tool_call(
                    agent=agent,
                    tool_name=tool_name,
                    args=safe_args,
                    result=result if result is not None else "",
                    duration_ms=duration_ms,
                    error=error,
                )
            except Exception as trace_err:
                logger.warning(f"Trace logging failed for {tool_name}: {trace_err}")

    return wrapper


# ==================== Pydantic Schema Factory ====================

def create_pydantic_schema(tool_config: Dict[str, Any]) -> Type[BaseModel]:
    """Create a Pydantic schema class from tool configuration.

    Args:
        tool_config: Tool configuration dictionary with 'parameters' key

    Returns:
        Pydantic BaseModel class for the tool
    """
    fields = {}
    parameters = tool_config.get('parameters', {})

    for param_name, param_info in parameters.items():
        field_type = _get_python_type(param_info.get('type', 'string'))
        field_default = param_info.get('default', ...)

        description = param_info.get('description', '')
        if param_info.get('required', True):
            fields[param_name] = (field_type, Field(description=description))
        else:
            fields[param_name] = (Optional[field_type], Field(default=field_default, description=description))

    return type(f'{tool_config["name"]}Input', (BaseModel,), fields)


def _get_python_type(type_str: str) -> type:
    """Convert string type to Python type."""
    type_map = {
        'string': str,
        'str': str,
        'integer': int,
        'int': int,
        'float': float,
        'number': float,
        'boolean': bool,
        'bool': bool,
        'array': List,
        'list': List,
        'object': Dict,
    }
    return type_map.get(type_str.lower(), str)


# ==================== Predefined Tool Schemas ====================

class CsvReaderInput(BaseModel):
    """Input schema for csv_reader_tool"""
    data_path: str = Field(..., description="Path to the CSV file (e.g., data/ZH_dataset/0105/data.csv or ./data/sample.csv)")


class RcdToolInput(BaseModel):
    """Input schema for rcd_tool"""
    inject_time: Union[float, str] = Field(
        ...,
        description="故障注入时间。支持 datetime 字符串（如 '2026-01-05 05:48:00'）或 Unix 时间戳（秒）。推荐直接传入字符串，无需手动计算时间戳。"
    )
    gamma: int = Field(default=5, description="Gamma parameter for IAF-RCL algorithm (default: 5)")
    abnormal_kpi: Optional[str] = Field(default=None, description="Name of the abnormal KPI metric (optional)")


class PcToolInput(BaseModel):
    """Input schema for pc_tool"""
    alpha: float = Field(default=0.05, description="Significance level for independence tests (default: 0.05)")
    abnormal_kpi: Optional[str] = Field(default=None, description="Name of the abnormal KPI metric (optional)")


class ThreeSigmaInput(BaseModel):
    """Input schema for three_sigma_tool"""
    inject_time: Union[float, str] = Field(
        ...,
        description="故障注入时间。支持 datetime 字符串（如 '2026-01-05 05:48:00'）或 Unix 时间戳（秒）。推荐直接传入字符串，无需手动计算时间戳。"
    )
    baseline_start_minutes: int = Field(default=30, description="基线窗口起始偏移（分钟），基线窗口为 [inject_time - baseline_end, inject_time - baseline_start)（默认 30）")
    baseline_end_minutes: int = Field(default=60, description="基线窗口结束偏移（分钟），基线窗口为 [inject_time - baseline_end, inject_time - baseline_start)（默认 60）")
    detect_before_minutes: int = Field(default=10, description="检测窗口前展长度（分钟），故障注入时间前 N 分钟纳入检测（默认 10）")
    detect_minutes: int = Field(default=10, description="检测窗口后延长度（分钟），故障注入时间后 N 分钟纳入检测（默认 10）")
    threshold: float = Field(default=3.0, description="Number of standard deviations for anomaly threshold (default: 3.0)")
    metric_columns: Optional[List[str]] = Field(default=None, description="Specific metric columns to check; if None, all numeric columns except 'time'")


class GraphVisualizationToolInput(BaseModel):
    """Input schema for graph_visualization_tool"""
    edges: List[List[str]] = Field(..., description="List of directed edges [[source, target], ...]")
    root_causes: List[str] = Field(..., description="List of root cause metrics")
    abnormal_kpi: Optional[str] = Field(default=None, description="The abnormal KPI metric name (optional)")
    output_format: str = Field(default="html", description="Output format: 'html' (interactive page)")
    output_dir: str = Field(default="outputs/graphs", description="Directory to save output files")


class AskUserInput(BaseModel):
    """Input schema for ask_user tool"""
    question: str = Field(..., description="Question to ask the user")
    tool: Optional[str] = Field(default=None, description="Name of the tool that needs the input")


# ==================== LangChain Tool Factory ====================

def create_langchain_tools(
    tool_names: List[str],
    tool_registry: Optional[ToolRegistry] = None
) -> List[StructuredTool]:
    """Create LangChain StructuredTool instances from tool names.

    Args:
        tool_names: List of tool names to create
        tool_registry: ToolRegistry instance (creates default if None)

    Returns:
        List of LangChain StructuredTool instances
    """
    if tool_registry is None:
        tool_registry = ToolRegistry()
        tool_registry.load_tools()

    tools = []

    for tool_name in tool_names:
        tool_func = tool_registry.get_tool_function(tool_name)
        if tool_func is None:
            logger.warning(f"Tool function not found: {tool_name}")
            continue

        tool_config = tool_registry.tools.get(tool_name, {})
        description = tool_config.get("description", "")

        # Create wrapped function that returns JSON string
        if tool_name == "csv_reader_tool":
            wrapped_func = _wrap_csv_reader(tool_func)
            args_schema = CsvReaderInput
        elif tool_name == "three_sigma_tool":
            wrapped_func = _wrap_three_sigma(tool_func)
            args_schema = ThreeSigmaInput
        elif tool_name == "rcd_tool":
            wrapped_func = _wrap_rcd_tool(tool_func)
            args_schema = RcdToolInput
        elif tool_name == "pc_tool":
            wrapped_func = _wrap_pc_tool(tool_func)
            args_schema = PcToolInput
        elif tool_name == "graph_visualization_tool":
            wrapped_func = _wrap_graph_visualization(tool_func)
            args_schema = GraphVisualizationToolInput
        else:
            # Generic wrapper for other tools
            def generic_wrapper(**kwargs):
                try:
                    result = tool_func(**kwargs)
                    if isinstance(result, dict):
                        return json.dumps(result, ensure_ascii=False)
                    return str(result)
                except Exception as e:
                    return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)

            wrapped_func = generic_wrapper
            args_schema = create_pydantic_schema(tool_config) if tool_config else None

        # Wrap with trace logging
        wrapped_func = _trace_tool_wrapper(tool_name, wrapped_func)

        tool = StructuredTool.from_function(
            func=wrapped_func,
            name=tool_name,
            description=description,
            args_schema=args_schema
        )
        tools.append(tool)
        logger.info(f"Created LangChain tool: {tool_name}")

    return tools


def create_diagnose_tools(tool_registry: Optional[ToolRegistry] = None) -> List[StructuredTool]:
    """Create the standard set of tools for diagnose agent.

    Args:
        tool_registry: ToolRegistry instance (creates default if None)

    Returns:
        List of LangChain StructuredTool instances for diagnose agent
    """
    tool_names = ["csv_reader_tool", "rcd_tool", "pc_tool", "graph_visualization_tool"]
    return create_langchain_tools(tool_names, tool_registry)
