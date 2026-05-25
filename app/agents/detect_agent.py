"""Detect Agent - Sequential workflow for anomaly detection

Fixed sequential workflow:
1. parse_params   — extract parameters from user request (LLM)
2. load_csv       — read and cache the CSV data
3. run_three_sigma — 3-sigma anomaly detection
4. refine         — LLM synthesizes detection report

Graph structure:
START → parse_params → load_csv → run_three_sigma → refine → END
"""

from typing import Dict, Any, List, Optional, TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langgraph.graph.ui import AnyUIMessage, ui_message_reducer
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
import pandas as pd
import json
import re
import uuid

from app.config.model_config import get_deepseek_llm
from app.utils.logger import get_logger
from app.utils.prompt_loader import load_prompt
from app.utils.llm_logger import log_llm_conversation
from app.tools.three_sigma import run_three_sigma
from app.utils.path_resolver import resolve_data_path

logger = get_logger(__name__)


def add_messages(left: List[BaseMessage], right: List[BaseMessage]) -> List[BaseMessage]:
    if not right:
        return left
    return left + right


# ==================== State Definition ====================

class DetectAgentState(TypedDict, total=False):
    """State for sequential detect agent workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    ui: Annotated[Sequence[AnyUIMessage], ui_message_reducer]
    task_description: str
    csv_file_path: Optional[str]
    inject_time: Optional[float]
    csv_headers: Optional[List[str]]
    three_sigma_result: Optional[Dict[str, Any]]
    tool_errors: List[Dict[str, Any]]
    integrated_result: Optional[str]


# ==================== CSV Data Cache ====================

_csv_data_cache: Dict[str, pd.DataFrame] = {}


def get_cached_csv(data_path: str) -> Optional[pd.DataFrame]:
    return _csv_data_cache.get(data_path)


def cache_csv_data(data_path: str, df: pd.DataFrame) -> None:
    _csv_data_cache[data_path] = df


# ==================== Helper Functions ====================

def _normalize_time_column(df: pd.DataFrame) -> pd.DataFrame:
    """Convert time column to Unix timestamp (seconds) if it contains datetime strings."""
    if 'time' not in df.columns:
        return df

    time_col = df['time']
    if pd.api.types.is_numeric_dtype(time_col):
        return df

    try:
        time_series = pd.to_datetime(time_col)
        time_ns = time_series.astype('datetime64[ns]').astype('int64')
        df = df.copy()
        df['time'] = time_ns // 10**9
        logger.info(
            f"Time column converted: datetime → Unix timestamp "
            f"(range: {df['time'].min()} — {df['time'].max()})"
        )
    except Exception as e:
        logger.warning(f"Time column is not numeric and could not be parsed as datetime: {e}")

    return df


def _parse_inject_time(inject_time) -> Optional[float]:
    """Parse inject_time to a numeric Unix timestamp."""
    if inject_time is None:
        return None
    if isinstance(inject_time, (int, float)):
        return float(inject_time)
    if isinstance(inject_time, str) and inject_time.strip():
        try:
            return pd.Timestamp(inject_time.strip()).timestamp()
        except Exception:
            try:
                return float(inject_time.strip())
            except ValueError:
                return None
    return None


def _make_tool_message(tool_name: str, result: Dict[str, Any], tool_call_id: str = None) -> list:
    """Create an AIMessage with synthetic tool_calls + a ToolMessage pair."""
    call_id = tool_call_id or f"call_{uuid.uuid4().hex[:12]}"
    ai_msg = AIMessage(
        content="",
        tool_calls=[{
            "name": tool_name,
            "args": {k: v for k, v in result.get("args", {}).items() if v is not None},
            "id": call_id,
            "type": "tool_call",
        }]
    )
    summary_keys = {"success", "status", "error", "anomalies", "shape",
                    "columns", "data_path", "message", "algorithm"}
    tool_content = {k: v for k, v in result.items() if k in summary_keys and k != "args"}
    tool_msg = ToolMessage(
        content=json.dumps(tool_content, ensure_ascii=False, indent=2),
        tool_call_id=call_id,
        name=tool_name
    )
    return [ai_msg, tool_msg]


# ==================== Graph Nodes ====================

_PARSE_PARAMS_PROMPT = """从用户请求中提取异常检测所需的参数，返回严格的 JSON 格式。

## 数据文件路径规则

文件路径格式为 data/ZH_dataset/{MMDD}/data.csv，其中 MMDD 是从故障日期提取的月日（两位，不足补零）：
- "2026年1月5日" → MMDD="0105" → data_path="data/ZH_dataset/0105/data.csv"
- "11月16日"       → MMDD="1116" → data_path="data/ZH_dataset/1116/data.csv"

注意：1月必须补零为01，5日必须补零为05！

## 需要提取的参数

1. **data_path** (string, 必需) — 按上述规则构造的 CSV 文件路径
2. **inject_time** (string|null) — 故障发生时间。格式 "YYYY-MM-DD HH:MM:SS"

## 输出格式

只输出 JSON：
```json
{
  "data_path": "data/ZH_dataset/0105/data.csv",
  "inject_time": "2026-01-05 05:48:00"
}
```"""


def _build_path_from_task(task: str) -> Optional[str]:
    """Extract date from task and construct data/ZH_dataset/{MMDD}/data.csv."""
    # Pattern 1: Chinese date — "2026年1月5日" or "1月5日"
    m = re.search(r'(\d{1,2})\s*月\s*(\d{1,2})\s*日', task)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"data/ZH_dataset/{month:02d}{day:02d}/data.csv"

    # Pattern 2: MM-DD — "04-05" or "4-5"
    m = re.search(r'\b(\d{1,2})-(\d{1,2})\b', task)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"data/ZH_dataset/{month:02d}{day:02d}/data.csv"

    # Pattern 3: already MMDD — "0105"
    m = re.search(r'(?:ZH_dataset/)(\d{4})', task)
    if m:
        return f"data/ZH_dataset/{m.group(1)}/data.csv"

    return None


def parse_params_node(state: DetectAgentState) -> DetectAgentState:
    """Parse parameters: LLM extracts data_path and inject_time, regex fallback."""
    task = state.get("task_description", "")
    logger.info(f"DETECT: Parsing parameters from: {task[:100]}...")

    if "tool_errors" not in state:
        state["tool_errors"] = []

    parsed_params = {}

    if not task or not task.strip():
        state["tool_errors"].append({"step": "parse_params", "error": "task_description 为空，无法提取参数"})
        state["messages"] = _make_tool_message("parse_params", {
            "success": False,
            "error": "task_description 为空",
            "args": {},
        })
        return state

    try:
        llm = get_deepseek_llm(temperature=0)
        response = llm.invoke(task, system_prompt=_PARSE_PARAMS_PROMPT).strip()
        logger.info(f"DETECT: LLM parse response: {response}")

        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            params = json.loads(json_match.group(0))
        else:
            logger.warning(f"DETECT: LLM did not return JSON: {response[:200]}")
            params = {}

        # Set csv_file_path
        csv_path = params.get("data_path", "")
        if csv_path and csv_path.strip().endswith(".csv"):
            state["csv_file_path"] = csv_path.strip()
            parsed_params["data_path"] = csv_path.strip()
            logger.info(f"DETECT: LLM data_path: {csv_path}")
        else:
            csv_path = _build_path_from_task(task)
            if csv_path:
                state["csv_file_path"] = csv_path
                parsed_params["data_path"] = csv_path
                logger.info(f"DETECT: Fallback data_path: {csv_path}")
            else:
                state["tool_errors"].append({"step": "parse_params", "error": "未找到有效的 CSV 文件路径"})

        # Set inject_time
        inject_time_raw = params.get("inject_time")
        if inject_time_raw is not None and inject_time_raw != "":
            parsed = _parse_inject_time(inject_time_raw)
            if parsed is not None:
                state["inject_time"] = parsed
                parsed_params["inject_time"] = inject_time_raw
                logger.info(f"DETECT: Extracted inject_time: {inject_time_raw} → {parsed}")

    except Exception as e:
        logger.error(f"DETECT: LLM parsing failed: {e}")
        csv_path = _build_path_from_task(task)
        if csv_path:
            state["csv_file_path"] = csv_path
            parsed_params["data_path"] = csv_path

    state["messages"] = _make_tool_message("parse_params", {
        "success": bool(state.get("csv_file_path")),
        "args": parsed_params,
        "csv_file_path": state.get("csv_file_path"),
        "inject_time": state.get("inject_time"),
    })

    return state


def load_csv_node(state: DetectAgentState) -> DetectAgentState:
    """Read the CSV data file and cache it."""
    csv_path = state.get("csv_file_path")
    if not csv_path:
        logger.error("DETECT: No csv_file_path to load")
        state["tool_errors"] = state.get("tool_errors", []) + [{
            "step": "load_csv",
            "error": "未提供 CSV 文件路径"
        }]
        state["messages"] = _make_tool_message("load_csv", {
            "success": False,
            "error": "未提供 CSV 文件路径",
            "args": {"data_path": csv_path}
        })
        return state

    logger.info(f"DETECT: Loading CSV from: {csv_path}")

    try:
        resolved_path = resolve_data_path(data_path=csv_path)
        if not resolved_path:
            resolved_path = csv_path

        df = pd.read_csv(resolved_path)

        dup_cols = df.columns[df.columns.duplicated()].unique().tolist()
        if dup_cols:
            logger.warning(f"Removing {len(dup_cols)} duplicate column(s): {dup_cols}")
            df = df.loc[:, ~df.columns.duplicated()]

        df = _normalize_time_column(df)

        state["csv_headers"] = df.columns.tolist()
        state["csv_file_path"] = resolved_path
        cache_csv_data(resolved_path, df)

        logger.info(f"DETECT: CSV loaded — shape={df.shape}")

        state["messages"] = _make_tool_message("load_csv", {
            "success": True,
            "args": {"data_path": resolved_path},
            "data_path": resolved_path,
            "shape": [df.shape[0], df.shape[1]],
            "columns": df.columns.tolist()[:20],
            "time_range": f"{df['time'].min()} — {df['time'].max()}" if "time" in df.columns else "N/A",
        })

    except Exception as e:
        logger.error(f"DETECT: CSV load failed: {e}")
        state["tool_errors"] = state.get("tool_errors", []) + [{
            "step": "load_csv",
            "error": str(e)
        }]
        state["messages"] = _make_tool_message("load_csv", {
            "success": False,
            "error": str(e),
            "args": {"data_path": csv_path}
        })

    return state


def run_three_sigma_node(state: DetectAgentState) -> DetectAgentState:
    """Run 3-sigma anomaly detection on loaded CSV data."""
    csv_path = state.get("csv_file_path", "")
    df = get_cached_csv(csv_path)
    inject_time = state.get("inject_time")

    if not inject_time:
        logger.info("DETECT: No inject_time, skipping 3-sigma")
        state["messages"] = _make_tool_message("three_sigma_tool", {
            "success": False,
            "error": "未提供 inject_time，无法执行 3-sigma 异常检测",
            "args": {"inject_time": None}
        })
        state["tool_errors"] = state.get("tool_errors", []) + [{
            "step": "three_sigma",
            "error": "未提供 inject_time"
        }]
        return state

    if df is None:
        logger.error("DETECT: No CSV data for 3-sigma")
        state["tool_errors"] = state.get("tool_errors", []) + [{
            "step": "three_sigma",
            "error": "CSV 数据未加载"
        }]
        return state

    logger.info(f"DETECT: Running 3-sigma — inject_time={inject_time}")

    try:
        result_json = run_three_sigma(data=df, inject_time=inject_time)
        result = json.loads(result_json)
        result["success"] = result.get("success", False)
        state["three_sigma_result"] = result

        anomalies = result.get("anomalies", [])
        logger.info(f"DETECT: 3-sigma done — {len(anomalies)} anomalous metrics")

        state["messages"] = _make_tool_message("three_sigma_tool", {
            "success": result.get("success", False),
            "args": {"inject_time": inject_time},
            "metrics_checked": result.get("metrics_checked", 0),
            "anomalies_found": len(anomalies),
            "top_anomalies": anomalies[:10],
        })

    except Exception as e:
        logger.error(f"DETECT: 3-sigma failed: {e}")
        state["three_sigma_result"] = {"success": False, "error": str(e), "anomalies": []}
        state["tool_errors"] = state.get("tool_errors", []) + [{"step": "three_sigma", "error": str(e)}]
        state["messages"] = _make_tool_message("three_sigma_tool", {
            "success": False,
            "error": str(e),
            "args": {"inject_time": inject_time},
        })

    return state


def refine_node(state: DetectAgentState) -> DetectAgentState:
    """LLM synthesizes anomaly detection report from 3-sigma results."""
    logger.info("DETECT: Generating detection report")

    three_sigma = state.get("three_sigma_result")
    tool_errors = state.get("tool_errors", [])
    task_description = state.get("task_description", "")

    parts = []

    if task_description:
        parts.append(f"## 任务描述\n{task_description}")

    inject_time = state.get("inject_time")
    if inject_time:
        from datetime import datetime, timezone, timedelta
        tz = timezone(timedelta(hours=8))
        dt_str = datetime.fromtimestamp(inject_time, tz=tz).strftime("%Y-%m-%d %H:%M:%S")
        parts.append(f"\n## 检测参数\n- 故障注入时间: {dt_str} (Unix: {inject_time})")

    csv_path = state.get("csv_file_path", "")
    df = get_cached_csv(csv_path)
    if df is not None:
        parts.append(f"\n## CSV 数据\n- 文件: {csv_path}\n- 形状: {df.shape[0]} 行 × {df.shape[1]} 列")
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        parts.append(f"- 数值指标 ({len(num_cols)} 个): {', '.join(num_cols[:30])}")

    if three_sigma:
        parts.append("\n## 3-Sigma 异常检测结果")
        if three_sigma.get("success"):
            anomalies = three_sigma.get("anomalies", [])
            params = three_sigma.get("parameters", {})
            parts.append(f"- 状态: 成功")
            parts.append(f"- 基线/检测窗口: {params.get('baseline_minutes', '?')}min / {params.get('detect_minutes', '?')}min")
            parts.append(f"- 阈值: {params.get('threshold', '?')}σ")
            parts.append(f"- 扫描指标数: {three_sigma.get('metrics_checked', '?')}")
            parts.append(f"- 异常指标数: {len(anomalies)}")
            if anomalies:
                parts.append("- 异常指标 (按z-score降序):")
                for a in anomalies[:15]:
                    parts.append(
                        f"  - {a['metric']}: z={a['z_score']:.2f}, "
                        f"value={a['value']:.4f}, baseline μ={a['baseline_mean']:.4f}±{a['baseline_std']:.4f}"
                    )
        else:
            parts.append(f"- 状态: 失败\n- 错误: {three_sigma.get('error', 'Unknown')}")

    if tool_errors:
        parts.append("\n## 执行错误")
        for err in tool_errors:
            parts.append(f"- {err.get('step', 'unknown')}: {err.get('error', 'unknown')}")

    result_str = "\n".join(parts)

    try:
        refine_prompt = load_prompt("app/prompts/detect_refine.md")
        prompt = refine_prompt.replace("{{RESULT_STR}}", result_str)
    except Exception:
        if three_sigma and three_sigma.get("success"):
            prompt = (
                f"你是一个 AIOps 异常检测专家。基于以下 3-Sigma 检测结果生成报告：\n\n{result_str}\n\n"
                f"请生成包含异常检测概况、异常指标列表、异常模式分析、结论与建议的结构化报告。"
            )
        else:
            prompt = f"未能成功执行异常检测。\n\n任务: {task_description}\n\n请说明问题并给出建议。"

    try:
        llm = get_deepseek_llm(temperature=0.3)
        final_result = llm.invoke(prompt)
        final_content = final_result if isinstance(final_result, str) else str(final_result)

        log_llm_conversation(
            agent_name="detect_refine",
            iteration=1,
            input_messages=[HumanMessage(content=prompt)],
            response=AIMessage(content=final_content),
            metadata={
                "three_sigma_executed": bool(three_sigma),
                "tool_errors_count": len(tool_errors),
                "type": "final_detection_report"
            }
        )

        state["integrated_result"] = final_content
    except Exception as e:
        logger.error(f"DETECT: Refine failed: {e}")
        state["integrated_result"] = result_str + "\n\n注: LLM报告生成失败，以上为原始检测结果。"

    # Chat: report as markdown
    state["messages"] = [AIMessage(content=state["integrated_result"])]

    logger.info("DETECT: Detection report generated")
    return state


# ==================== Graph Builder ====================

def build_detect_agent() -> StateGraph:
    """Build the detect agent with sequential workflow.

    Graph structure:
    START → parse_params → load_csv → run_three_sigma → refine → END
    """
    logger.info("Building detect agent with sequential workflow")

    builder = StateGraph(DetectAgentState)

    builder.add_node("parse_params", parse_params_node)
    builder.add_node("load_csv", load_csv_node)
    builder.add_node("run_three_sigma", run_three_sigma_node)
    builder.add_node("refine", refine_node)

    builder.set_entry_point("parse_params")
    builder.add_edge("parse_params", "load_csv")
    builder.add_edge("load_csv", "run_three_sigma")
    builder.add_edge("run_three_sigma", "refine")
    builder.add_edge("refine", END)

    graph = builder.compile()
    logger.info("Detect agent compiled with sequential workflow")
    return graph


# Create global instance
detect_agent = build_detect_agent()
