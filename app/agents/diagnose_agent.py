"""Diagnose Agent - Sequential workflow for root cause analysis

Fixed sequential workflow:
1. parse_params   — extract parameters from user request (LLM)
2. load_csv       — read and cache the CSV data
3. run_three_sigma — 3-sigma anomaly pre-filter (if inject_time available)
4. run_rcd        — execute IAF-RCL algorithm (if inject_time available)
5. run_pc         — execute KE-FPC algorithm
6. visualize_graph — build propagation topology HTML
7. refine         — LLM synthesizes final report

Each node writes AIMessage.tool_calls + ToolMessage pairs to the messages
state so LangGraph Studio can visualize the tool execution in its trace view.

Graph structure:
START → parse_params → load_csv → run_rcd → run_pc → visualize_graph → refine → END
"""

from typing import Dict, Any, List, Optional, Union, TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langgraph.graph.ui import AnyUIMessage, ui_message_reducer
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
import pandas as pd
import json
import re
import uuid

from app.config.algorithm_names import ALGORITHM_IAF_RCL, ALGORITHM_KE_FPC
from app.config.model_config import get_deepseek_llm
from app.utils.logger import get_logger
from app.utils.prompt_loader import load_prompt
from app.utils.llm_logger import log_llm_conversation
from app.tools.rcd_wrapper import run_rcd_analysis
from app.tools.pc_wrapper import run_pc_analysis
from app.tools.three_sigma import run_three_sigma
from app.tools.graph_visualization_tool import visualize_propagation_graph, get_topology_view_url
from app.utils.topology_chat import (
    build_final_report_messages,
    format_propagation_path_line,
)
from app.utils.json_utils import sanitize_for_json
from app.utils.path_resolver import resolve_data_path

logger = get_logger(__name__)


def add_messages(left: List[BaseMessage], right: List[BaseMessage]) -> List[BaseMessage]:
    """Add messages reducer for langgraph state"""
    if not right:
        return left
    return left + right


# ==================== State Definition ====================

class DiagnoseAgentState(TypedDict, total=False):
    """State for sequential diagnose agent workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    ui: Annotated[Sequence[AnyUIMessage], ui_message_reducer]
    task_description: str
    # Parsed parameters
    csv_file_path: Optional[str]
    inject_time: Optional[float]
    abnormal_kpi: Optional[str]
    gamma: int
    alpha: float
    # Results (csv_data is held in module-level _csv_data_cache, not in state)
    csv_headers: Optional[List[str]]
    rcd_result: Optional[Dict[str, Any]]
    three_sigma_result: Optional[Dict[str, Any]]
    pc_result: Optional[Dict[str, Any]]
    graph_visualization: Optional[Dict[str, Any]]
    detect_result: Optional[Dict[str, Any]]
    tool_errors: List[Dict[str, Any]]
    integrated_result: Optional[str]


# ==================== CSV Data Cache ====================

_csv_data_cache: Dict[str, pd.DataFrame] = {}


def get_cached_csv(data_path: str) -> Optional[pd.DataFrame]:
    """Get cached CSV data for a path."""
    return _csv_data_cache.get(data_path)


# ==================== Helper Functions ====================

def _normalize_time_column(df: pd.DataFrame) -> pd.DataFrame:
    """Convert time column to Unix timestamp (seconds) if it contains datetime strings."""
    if 'time' not in df.columns:
        return df

    time_col = df['time']
    if pd.api.types.is_numeric_dtype(time_col):
        logger.info(f"Time column is numeric (min={time_col.min()}, max={time_col.max()})")
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


def cache_csv_data(data_path: str, df: pd.DataFrame) -> None:
    """Cache CSV data for a path."""
    _csv_data_cache[data_path] = df


def _make_tool_message(tool_name: str, result: Dict[str, Any], tool_call_id: str = None) -> list:
    """Create an AIMessage with synthetic tool_calls + a ToolMessage pair.

    Returns a list of [AIMessage, ToolMessage] to append to state messages.
    This makes tool execution visible in LangGraph Studio's trace view.
    """
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
    # Build tool result summary for display
    summary_keys = {"success", "status", "error", "root_causes", "edges", "shape",
                    "columns", "data_path", "message", "algorithm"}
    tool_content = {k: v for k, v in result.items() if k in summary_keys and k != "args"}
    tool_msg = ToolMessage(
        content=json.dumps(tool_content, ensure_ascii=False, indent=2),
        tool_call_id=call_id,
        name=tool_name
    )
    return [ai_msg, tool_msg]


# ==================== Graph Nodes ====================

_PARSE_PARAMS_PROMPT = """从用户请求中提取根因分析所需的参数，返回严格的 JSON 格式。

## 数据文件路径规则

文件路径格式为 data/ZH_dataset/{MMDD}/data.csv，其中 MMDD 是从故障日期提取的月日（两位，不足补零）：
- "2026年1月5日" → MMDD="0105" → data_path="data/ZH_dataset/0105/data.csv"
- "11月16日"       → MMDD="1116" → data_path="data/ZH_dataset/1116/data.csv"

注意：1月必须补零为01，5日必须补零为05！

## 需要提取的参数

1. **data_path** (string, 必需) — 按上述规则构造的 CSV 文件路径
2. **inject_time** (string|null) — 故障发生时间。格式 "YYYY-MM-DD HH:MM:SS"
3. **abnormal_kpi** (string|null) — 异常指标名称。若未提及则返回 null

## 输出格式

只输出 JSON：
```json
{
  "data_path": "data/ZH_dataset/0105/data.csv",
  "inject_time": "2026-01-05 05:48:00",
  "abnormal_kpi": "full_request_duration_ms_new_10.104.128.205:9093"
}
```"""


def parse_params_node(state: DiagnoseAgentState) -> DiagnoseAgentState:
    """Parse parameters: LLM extracts all params, regex fallback for data_path."""
    task = state.get("task_description", "")
    logger.info(f"DIAGNOSE: Parsing parameters from: {task[:100]}...")

    if "tool_errors" not in state:
        state["tool_errors"] = []

    parsed_params = {}

    # Guard: don't call LLM with empty input — it will hallucinate from the prompt examples
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
        logger.info(f"DIAGNOSE: LLM parse response: {response}")

        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            params = json.loads(json_match.group(0))
        else:
            logger.warning(f"DIAGNOSE: LLM did not return JSON: {response[:200]}")
            params = {}

        # --- Set csv_file_path (LLM primary, regex fallback) ---
        csv_path = params.get("data_path", "")
        if csv_path and csv_path.strip().endswith(".csv"):
            state["csv_file_path"] = csv_path.strip()
            parsed_params["data_path"] = csv_path.strip()
            logger.info(f"DIAGNOSE: LLM data_path: {csv_path}")
        else:
            # Fallback: regex date extraction
            csv_path = _build_path_from_task(task)
            if csv_path:
                state["csv_file_path"] = csv_path
                parsed_params["data_path"] = csv_path
                logger.info(f"DIAGNOSE: Fallback data_path: {csv_path}")
            else:
                state["tool_errors"].append({"step": "parse_params", "error": "未找到有效的 CSV 文件路径"})

        # --- Set inject_time ---
        inject_time_raw = params.get("inject_time")
        if inject_time_raw is not None and inject_time_raw != "":
            parsed = _parse_inject_time(inject_time_raw)
            if parsed is not None:
                state["inject_time"] = parsed
                parsed_params["inject_time"] = inject_time_raw
                logger.info(f"DIAGNOSE: Extracted inject_time: {inject_time_raw} → {parsed}")

        # --- Set abnormal_kpi ---
        abnormal_kpi = params.get("abnormal_kpi")
        if abnormal_kpi and abnormal_kpi != "":
            state["abnormal_kpi"] = abnormal_kpi
            parsed_params["abnormal_kpi"] = abnormal_kpi
            logger.info(f"DIAGNOSE: Extracted abnormal_kpi: {abnormal_kpi}")

    except Exception as e:
        logger.error(f"DIAGNOSE: LLM parsing failed: {e}")
        # Fallback: try regex for path
        csv_path = _build_path_from_task(task)
        if csv_path:
            state["csv_file_path"] = csv_path
            parsed_params["data_path"] = csv_path

    # Record tool call in messages for Studio visibility
    state["messages"] = _make_tool_message("parse_params", {
        "success": bool(state.get("csv_file_path")),
        "args": parsed_params,
        "csv_file_path": state.get("csv_file_path"),
        "inject_time": state.get("inject_time"),
        "abnormal_kpi": state.get("abnormal_kpi"),
    })

    return state


def _build_path_from_task(task: str) -> Optional[str]:
    """Extract date from task and construct data/ZH_dataset/{MMDD}/data.csv.

    Handles formats: "11月16日", "2026年1月5日", "01-05", "0105"
    """
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


def load_csv_node(state: DiagnoseAgentState) -> DiagnoseAgentState:
    """Read the CSV data file and cache it."""
    csv_path = state.get("csv_file_path")
    if not csv_path:
        logger.error("DIAGNOSE: No csv_file_path to load")
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

    logger.info(f"DIAGNOSE: Loading CSV from: {csv_path}")

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

        logger.info(f"DIAGNOSE: CSV loaded — shape={df.shape}")

        state["messages"] = _make_tool_message("load_csv", {
            "success": True,
            "args": {"data_path": resolved_path},
            "data_path": resolved_path,
            "shape": [df.shape[0], df.shape[1]],
            "columns": df.columns.tolist()[:20],
            "time_range": f"{df['time'].min()} — {df['time'].max()}" if "time" in df.columns else "N/A",
        })

    except Exception as e:
        logger.error(f"DIAGNOSE: CSV load failed: {e}")
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


def run_three_sigma_node(state: DiagnoseAgentState) -> DiagnoseAgentState:
    """Run 3-sigma anomaly detection to pre-filter anomalous metrics."""
    csv_path = state.get("csv_file_path", "")
    df = get_cached_csv(csv_path)
    inject_time = state.get("inject_time")

    if not inject_time:
        logger.info("DIAGNOSE: No inject_time, skipping 3-sigma")
        state["messages"] = _make_tool_message("three_sigma_tool", {
            "success": False,
            "error": "未提供 inject_time，跳过 3-sigma 异常检测",
            "args": {"inject_time": None}
        })
        return state

    if df is None:
        logger.error("DIAGNOSE: No CSV data for 3-sigma")
        state["tool_errors"] = state.get("tool_errors", []) + [{
            "step": "three_sigma",
            "error": "CSV 数据未加载"
        }]
        return state

    logger.info(f"DIAGNOSE: Running 3-sigma — inject_time={inject_time}")

    try:
        result_json = run_three_sigma(data=df, inject_time=inject_time)
        result = json.loads(result_json)
        result["success"] = result.get("success", False)
        state["three_sigma_result"] = result

        anomalies = result.get("anomalies", [])
        logger.info(f"DIAGNOSE: 3-sigma done — {len(anomalies)} anomalous metrics")

        state["messages"] = _make_tool_message("three_sigma_tool", {
            "success": result.get("success", False),
            "args": {"inject_time": inject_time},
            "metrics_checked": result.get("metrics_checked", 0),
            "anomalies_found": len(anomalies),
            "top_anomalies": anomalies[:10],
        })

    except Exception as e:
        logger.error(f"DIAGNOSE: 3-sigma failed: {e}")
        state["three_sigma_result"] = {"success": False, "error": str(e), "anomalies": []}
        state["tool_errors"] = state.get("tool_errors", []) + [{"step": "three_sigma", "error": str(e)}]
        state["messages"] = _make_tool_message("three_sigma_tool", {
            "success": False,
            "error": str(e),
            "args": {"inject_time": inject_time},
        })

    return state


def run_rcd_node(state: DiagnoseAgentState) -> DiagnoseAgentState:
    """Run IAF-RCL algorithm if inject_time is available."""
    csv_path = state.get("csv_file_path", "")
    df = get_cached_csv(csv_path)
    inject_time = state.get("inject_time")

    if not inject_time:
        logger.info("DIAGNOSE: No inject_time, skipping RCD")
        state["messages"] = _make_tool_message("rcd_algorithm", {
            "success": False,
            "error": f"未提供 inject_time，跳过 {ALGORITHM_IAF_RCL}",
            "args": {"inject_time": None}
        })
        return state

    if df is None:
        logger.error("DIAGNOSE: No CSV data for RCD")
        state["tool_errors"] = state.get("tool_errors", []) + [{
            "step": "rcd",
            "error": "CSV 数据未加载"
        }]
        return state

    gamma = state.get("gamma", 5)
    abnormal_kpi = state.get("abnormal_kpi")

    logger.info(f"DIAGNOSE: Running RCD — inject_time={inject_time}, gamma={gamma}, "
                f"abnormal_kpi={abnormal_kpi}")

    try:
        result = run_rcd_analysis(
            data=df, inject_time=inject_time, gamma=gamma,
            localized=True, bins=5, dataset=None, verbose=False,
            abnormal_kpi=abnormal_kpi, seed=42
        )
        result["algorithm"] = ALGORITHM_IAF_RCL
        result["success"] = True
        state["rcd_result"] = result

        root_causes = result.get("root_causes", [])
        logger.info(f"DIAGNOSE: RCD done — {len(root_causes)} root causes")

        state["messages"] = _make_tool_message("rcd_algorithm", {
            "success": True,
            "args": {"inject_time": inject_time, "gamma": gamma, "abnormal_kpi": abnormal_kpi},
            "root_causes": root_causes[:10],
            "total_root_causes": len(root_causes),
            "algorithm": ALGORITHM_IAF_RCL,
        })

    except Exception as e:
        logger.error(f"DIAGNOSE: {ALGORITHM_IAF_RCL} failed: {e}")
        state["rcd_result"] = {"algorithm": ALGORITHM_IAF_RCL, "success": False, "error": str(e), "root_causes": []}
        state["tool_errors"] = state.get("tool_errors", []) + [{"step": "rcd", "error": str(e)}]
        state["messages"] = _make_tool_message("rcd_algorithm", {
            "success": False,
            "error": str(e),
            "args": {"inject_time": inject_time, "gamma": gamma},
        })

    return state


def run_pc_node(state: DiagnoseAgentState) -> DiagnoseAgentState:
    """Run KE-FPC algorithm for causal discovery."""
    csv_path = state.get("csv_file_path", "")
    df = get_cached_csv(csv_path)

    if df is None:
        logger.error("DIAGNOSE: No CSV data for PC")
        state["tool_errors"] = state.get("tool_errors", []) + [{
            "step": "pc",
            "error": "CSV 数据未加载"
        }]
        return state

    alpha = state.get("alpha", 0.05)
    abnormal_kpi = state.get("abnormal_kpi")

    logger.info(f"DIAGNOSE: Running PC — alpha={alpha}, abnormal_kpi={abnormal_kpi}")

    try:
        result = run_pc_analysis(
            data=df, alpha=alpha, verbose=False, abnormal_kpi=abnormal_kpi
        )
        result["algorithm"] = ALGORITHM_KE_FPC
        result["success"] = True
        state["pc_result"] = result

        rc = result.get("root_causes", [])
        edges = result.get("edges", [])
        logger.info(f"DIAGNOSE: PC done — {len(rc)} root causes, {len(edges)} edges")

        state["messages"] = _make_tool_message("pc_algorithm", {
            "success": True,
            "args": {"alpha": alpha, "abnormal_kpi": abnormal_kpi},
            "root_causes": rc[:10],
            "total_root_causes": len(rc),
            "edges": edges[:10],
            "total_edges": len(edges),
            "algorithm": ALGORITHM_KE_FPC,
        })

    except Exception as e:
        logger.error(f"DIAGNOSE: {ALGORITHM_KE_FPC} failed: {e}")
        state["pc_result"] = {"algorithm": ALGORITHM_KE_FPC, "success": False, "error": str(e), "root_causes": [], "edges": []}
        state["tool_errors"] = state.get("tool_errors", []) + [{"step": "pc", "error": str(e)}]
        state["messages"] = _make_tool_message("pc_algorithm", {
            "success": False,
            "error": str(e),
            "args": {"alpha": alpha, "abnormal_kpi": abnormal_kpi},
        })

    return state


def visualize_graph_node(state: DiagnoseAgentState) -> DiagnoseAgentState:
    """Generate interactive HTML topology for KE-FPC propagation chains."""
    pc_result = state.get("pc_result") or {}
    edges = pc_result.get("edges") or []
    root_causes = pc_result.get("root_causes") or []
    abnormal_kpi = state.get("abnormal_kpi")

    if not pc_result.get("success") or not edges:
        logger.info("DIAGNOSE: Skip topology visualization (no KE-FPC edges)")
        state["graph_visualization"] = {
            "success": False,
            "error": "无可用传播边，跳过拓扑图生成",
        }
        return state

    logger.info(f"DIAGNOSE: Building propagation topology — {len(edges)} edges")
    try:
        viz = visualize_propagation_graph(
            edges=edges,
            root_causes=root_causes,
            abnormal_kpi=abnormal_kpi,
            output_format="html",
            output_dir="outputs/graphs",
        )
        state["graph_visualization"] = viz

        state["messages"] = _make_tool_message("graph_visualization_tool", {
            "success": viz.get("success", False),
            "args": {
                "abnormal_kpi": abnormal_kpi,
                "edge_count": len(viz.get("edges", [])),
            },
            "view_url": viz.get("view_url"),
            "filepath": viz.get("filepath"),
            "propagation_paths": (viz.get("propagation_paths") or [])[:5],
        })
        logger.info(f"DIAGNOSE: Topology ready at {viz.get('view_url')}")
    except Exception as e:
        logger.error(f"DIAGNOSE: Topology visualization failed: {e}")
        state["graph_visualization"] = {"success": False, "error": str(e)}
        state["tool_errors"] = state.get("tool_errors", []) + [{"step": "visualize", "error": str(e)}]

    return state


def refine_node(state: DiagnoseAgentState) -> DiagnoseAgentState:
    """LLM synthesizes final report from all results."""
    logger.info("DIAGNOSE: Generating final report")

    rcd_result = state.get("rcd_result")
    pc_result = state.get("pc_result")
    tool_errors = state.get("tool_errors", [])
    task_description = state.get("task_description", "")

    parts = []

    if task_description:
        parts.append(f"## 任务描述\n{task_description}")

    inject_time = state.get("inject_time")
    abnormal_kpi = state.get("abnormal_kpi")
    if inject_time or abnormal_kpi:
        parts.append("\n## 分析参数")
        if inject_time:
            from datetime import datetime, timezone, timedelta
            tz = timezone(timedelta(hours=8))
            dt_str = datetime.fromtimestamp(inject_time, tz=tz).strftime("%Y-%m-%d %H:%M:%S")
            parts.append(f"- 故障注入时间: {dt_str} (Unix: {inject_time})")
        if abnormal_kpi:
            parts.append(f"- 异常指标: {abnormal_kpi}")

    csv_path = state.get("csv_file_path", "")
    headers = state.get("csv_headers")
    df = get_cached_csv(csv_path)
    if df is not None:
        parts.append(f"\n## CSV 数据\n- 文件: {csv_path}\n- 形状: {df.shape[0]} 行 × {df.shape[1]} 列")
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        parts.append(f"- 数值指标 ({len(num_cols)} 个): {', '.join(num_cols[:30])}")

    three_sigma = state.get("three_sigma_result")
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

    if rcd_result:
        parts.append(f"\n## {ALGORITHM_IAF_RCL} 算法结果")
        if rcd_result.get("success"):
            rc = rcd_result.get("root_causes", [])
            parts.append(f"- 状态: 成功\n- 根因数量: {len(rc)}")
            if rc:
                parts.append(f"- 根因列表 (前20): {rc[:20]}")
        else:
            parts.append(f"- 状态: 失败\n- 错误: {rcd_result.get('message', rcd_result.get('error', 'Unknown'))}")

    if pc_result:
        parts.append(f"\n## {ALGORITHM_KE_FPC} 算法结果")
        if pc_result.get("success"):
            rc = pc_result.get("root_causes", [])
            edges = pc_result.get("edges", [])
            parts.append(f"- 状态: 成功\n- 根因数量: {len(rc)}\n- 因果边: {len(edges)}")
            if rc:
                parts.append(f"- 根因列表 (前20): {rc[:20]}")
            if edges:
                parts.append(f"- 因果边 (前30): {edges[:30]}")
        else:
            parts.append(f"- 状态: 失败\n- 错误: {pc_result.get('message', pc_result.get('error', 'Unknown'))}")

    if tool_errors:
        parts.append("\n## 执行错误")
        for err in tool_errors:
            parts.append(f"- {err.get('step', 'unknown')}: {err.get('error', 'unknown')}")

    # Detect Agent independent result (for comparison when chained from call_diagnose)
    detect_result = state.get("detect_result")
    if detect_result:
        parts.append("\n## Detect Agent 独立检测结果（对比参考）")
        detect_ts = detect_result.get("three_sigma_result", {})
        if detect_ts and detect_ts.get("success"):
            detect_anomalies = detect_ts.get("anomalies", [])
            detect_params = detect_ts.get("parameters", {})
            parts.append(f"- 独立检测方法: 3-Sigma（基线: {detect_params.get('baseline_minutes', '?')}min / 检测: {detect_params.get('detect_minutes', '?')}min / 阈值: {detect_params.get('threshold', '?')}σ）")
            parts.append(f"- 独立扫描指标数: {detect_ts.get('metrics_checked', '?')}，检出异常: {len(detect_anomalies)}")
            if detect_anomalies:
                parts.append("- 独立检出异常指标 (按z-score降序):")
                for a in detect_anomalies[:10]:
                    parts.append(
                        f"  - {a['metric']}: z={a['z_score']:.2f}, "
                        f"value={a['value']:.4f}, baseline μ={a['baseline_mean']:.4f}±{a['baseline_std']:.4f}"
                    )
        elif detect_result.get("tool_errors"):
            parts.append(f"- 独立检测状态: 失败")
            for err in detect_result.get("tool_errors", []):
                parts.append(f"  - {err.get('step', 'unknown')}: {err.get('error', 'unknown')}")
        parts.append("\n请在报告中对比 Detect Agent 独立检测结果与 Diagnose Agent 内置 3-Sigma 结果，说明异同。")

    graph_viz = state.get("graph_visualization")
    if graph_viz and graph_viz.get("success"):
        view_url = graph_viz.get("view_url") or get_topology_view_url()
        parts.append("\n## 根因传播拓扑图")
        parts.append(f"- 交互式拓扑: {view_url}")
        paths = graph_viz.get("propagation_paths") or []
        if paths:
            parts.append(f"- 基于 {ALGORITHM_KE_FPC} 的传播路径（供你理解因果，勿在报告中重复列出）:")
            for i, path in enumerate(paths[:15], 1):
                parts.append(f"  - {format_propagation_path_line(i, path)}")
    elif graph_viz and graph_viz.get("error"):
        parts.append(f"\n## 根因传播拓扑图\n- 未生成: {graph_viz.get('error')}")

    result_str = "\n".join(parts)

    try:
        refine_prompt = load_prompt("app/prompts/diagnose_refine.md")
        prompt = refine_prompt.replace("{{RESULT_STR}}", result_str)
    except Exception:
        if rcd_result or pc_result:
            prompt = (
                f"你是一个 AIOps 根因分析专家。基于以下分析结果生成报告：\n\n{result_str}\n\n"
                f"请生成包含根因指标列表、故障传播路径、故障类型判断、结论与建议的结构化报告。"
                f"报告中算法名称必须使用 {ALGORITHM_IAF_RCL} 与 {ALGORITHM_KE_FPC}，"
                f"不要使用 RCD、PC 作为算法名称。"
            )
        else:
            prompt = f"未能成功执行任何根因分析算法。\n\n任务: {task_description}\n\n请说明问题并给出建议。"

    try:
        llm = get_deepseek_llm(temperature=0.3)
        final_result = llm.invoke(prompt)
        final_content = final_result if isinstance(final_result, str) else str(final_result)

        log_llm_conversation(
            agent_name="diagnose_refine",
            iteration=1,
            input_messages=[HumanMessage(content=prompt)],
            response=AIMessage(content=final_content),
            metadata={
                "rcd_executed": bool(rcd_result),
                "pc_executed": bool(pc_result),
                "tool_errors_count": len(tool_errors),
                "type": "final_refinement"
            }
        )

        state["integrated_result"] = final_content
    except Exception as e:
        logger.error(f"DIAGNOSE: Refine failed: {e}")
        state["integrated_result"] = result_str + "\n\n注: LLM报告生成失败，以上为原始执行结果。"

    # Chat: 第1条=拓扑(Markdown路径+Mermaid+交互链接)，第2条=LLM分析(纯文本章节)
    state["messages"] = build_final_report_messages(
        state["integrated_result"], graph_viz
    )

    logger.info("DIAGNOSE: Final report generated")
    return state


# ==================== Graph Builder ====================

def build_diagnose_agent() -> StateGraph:
    """Build the diagnose agent with sequential workflow.

    Graph structure:
    START → parse_params → load_csv → run_three_sigma → run_rcd → run_pc → visualize_graph → refine → END
    """
    logger.info("Building diagnose agent with sequential workflow")

    builder = StateGraph(DiagnoseAgentState)

    builder.add_node("parse_params", parse_params_node)
    builder.add_node("load_csv", load_csv_node)
    builder.add_node("run_three_sigma", run_three_sigma_node)
    builder.add_node("run_rcd", run_rcd_node)
    builder.add_node("run_pc", run_pc_node)
    builder.add_node("visualize_graph", visualize_graph_node)
    builder.add_node("refine", refine_node)

    builder.set_entry_point("parse_params")
    builder.add_edge("parse_params", "load_csv")
    builder.add_edge("load_csv", "run_three_sigma")
    builder.add_edge("run_three_sigma", "run_rcd")
    builder.add_edge("run_rcd", "run_pc")
    builder.add_edge("run_pc", "visualize_graph")
    builder.add_edge("visualize_graph", "refine")
    builder.add_edge("refine", END)

    graph = builder.compile()
    logger.info("Diagnose agent compiled with sequential workflow")
    return graph


# Create global instance
diagnose_agent = build_diagnose_agent()
