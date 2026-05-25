"""Supervisor Agent - Main orchestrator subgraph

This agent:
1. Receives user input via LLM
2. Determines user intent (diagnosis vs general chat)
3. Routes to diagnose subagent or responds directly
4. Does NOT validate parameters - this is handled by diagnose subagent

Graph structure:
┌─────────────────────────────────────────┐
│         Supervisor Agent Subgraph        │
│                                         │
│  ┌──────────────┐                       │
│  │supervisor_llm│                       │
│  │              │                       │
│  │ Intent:      │                       │
│  │ - Diagnose?  → call_diagnose        │
│  │ - Chat?      → respond              │
│  └──────────────┘                       │
│                                         │
│  No parameter extraction/validation      │
└─────────────────────────────────────────┘
"""

from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END, add_messages

from app.config.model_config import get_deepseek_llm
from app.utils.logger import get_logger
from app.utils.prompt_loader import load_prompt

logger = get_logger(__name__)


class SupervisorAgentState(TypedDict, total=False):
    """State for supervisor agent"""
    user_input: str
    messages: Annotated[List[BaseMessage], add_messages]
    action: str
    csv_file_path: Optional[str]
    inject_time: Optional[float]
    abnormal_kpi: Optional[str]
    gamma: Optional[int]
    alpha: Optional[float]
    dataset_type: Optional[str]
    diagnose_result: Optional[Dict[str, Any]]
    response_message: Optional[str]
    continue_conversation: bool


# Keywords that indicate diagnosis intent (conservative fallback when LLM fails)
DIAGNOSIS_KEYWORDS = [
    "异常事件", "故障注入", "根因分析", "根因定位", "故障诊断",
    "帮我分析", "请分析", "分析根因", "分析故障", "分析异常",
    "rcd算法", "pc算法", "iaf-rcl", "ke-fpc", "因果发现", "故障传播",
    "root cause", "diagnose anomaly",
]

# Keywords that indicate detection-only intent
DETECTION_KEYWORDS = [
    "异常检测", "检测异常", "3-sigma", "3sigma", "三西格玛",
    "异常扫描", "指标异常", "anomaly detection", "detect anomaly",
    "有哪些异常", "异常指标", "异常告警",
]

def is_diagnosis_intent(user_input: str) -> bool:
    """Conservative keyword match — only triggers on explicit diagnosis requests."""
    return any(kw in user_input.lower() for kw in DIAGNOSIS_KEYWORDS)

def is_detection_intent(user_input: str) -> bool:
    """Check if user intent is anomaly detection only (not full root cause analysis)."""
    user_lower = user_input.lower()
    has_detection = any(kw in user_lower for kw in DETECTION_KEYWORDS)
    has_diagnosis = any(kw in user_lower for kw in DIAGNOSIS_KEYWORDS)
    # Detection intent: has detection keywords but no diagnosis keywords
    return has_detection and not has_diagnosis


def supervisor_llm_node(state: SupervisorAgentState) -> SupervisorAgentState:
    """
    Supervisor LLM node - intent recognition and routing.

    Two cases:
    1. diagnose_result present → synthesize and respond
    2. Normal flow → intent recognition via LLM + keyword fallback

    NO parameter extraction or validation.
    Only determines: call_diagnose or respond.
    """
    user_input = state.get("user_input", "")
    messages = state.get("messages", [])
    diagnose_result = state.get("diagnose_result")

    logger.info(f"SUPERVISOR: Processing: {user_input[:100]}...")

    # Reset action
    state["action"] = "respond"

    # --- Case 1: Diagnose result available ---
    if diagnose_result:
        logger.info("SUPERVISOR: Processing diagnose result")
        integrated_result = diagnose_result.get("integrated_result")

        # If integrated_result is not set, build one from available information
        if not integrated_result:
            tool_errors = diagnose_result.get("tool_errors", [])
            if tool_errors:
                error_msgs = []
                for err in tool_errors:
                    if err.get("requires_user_input") and err.get("question"):
                        error_msgs.append(err["question"])
                    else:
                        step = err.get("step", err.get("tool", "unknown"))
                        error = err.get("error", "unknown error")
                        error_msgs.append(f"步骤 '{step}' 失败: {error}")

                if error_msgs:
                    integrated_result = "\n\n".join(error_msgs)
                else:
                    integrated_result = "分析过程中出现错误，请重试。"
            else:
                integrated_result = "根因分析已完成。"

        if not integrated_result:
            integrated_result = "根因分析已完成。"

        diagnose_ai = [
            m for m in (diagnose_result.get("messages") or [])
            if isinstance(m, AIMessage) and m.content
        ]
        state["messages"] = diagnose_ai if diagnose_ai else [AIMessage(content=integrated_result)]
        state["diagnose_result"] = None
        state["continue_conversation"] = True
        state["action"] = "respond"
        return state

    # --- Case 3: Normal intent recognition ---
    # Load system prompt from file (fallback to keyword-only if file missing)
    try:
        supervisor_system_prompt = load_prompt("app/prompts/supervisor_system.md")
    except Exception:
        logger.warning("SUPERVISOR: Could not load supervisor_system.md, using keyword fallback")
        if is_diagnosis_intent(user_input):
            state["action"] = "call_diagnose"
            state["continue_conversation"] = True
            return state
        else:
            state["messages"] = [AIMessage(content="您好！我是 AIOps 根因分析助手。请描述您的异常事件或提供数据文件，我将为您进行根因分析。")]
            state["continue_conversation"] = True
            state["action"] = "respond"
            return state

    # Get LLM decision for intent recognition
    llm = get_deepseek_llm(temperature=0.1)

    # Build conversation context
    context_parts = []
    for msg in messages[-5:]:
        if isinstance(msg, HumanMessage):
            context_parts.append(f"用户: {msg.content}")
        elif isinstance(msg, AIMessage):
            context_parts.append(f"助手: {msg.content}")

    context_parts.append(f"\n当前输入: {user_input}")
    context = "\n".join(context_parts)

    user_prompt = f"""分析用户输入并判断意图：

{context}

## 判断标准

如果用户请求以下操作，action 设为 "call_diagnose"（完整根因分析）：
- 微服务异常事件分析、故障根因定位、根因诊断
- 指标数据分析
- 使用 IAF-RCL/KE-FPC 算法
- 故障传播图生成、因果发现、故障传播路径分析
- **或者用户输入包含大量日志输出、技术术语或之前诊断结果的片段**（这表示用户在继续或询问之前的诊断工作）

如果用户仅请求异常检测（不涉及根因分析），action 设为 "call_detect"：
- 异常检测、3-Sigma 检测、三西格玛
- 扫描哪些指标异常、有哪些异常指标
- 指标异常告警

否则（简单的问候、帮助、闲聊等），action 设为 "respond"。

输出 JSON：
```json
{{
  "action": "call_diagnose | call_detect | respond",
  "message": "回复内容（仅在 respond 时）"
}}
```
"""

    try:
        response_text = llm.invoke(user_prompt, system_prompt=supervisor_system_prompt).strip()

        import json
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            decision = json.loads(json_match.group(0))
        else:
            # Fallback to keyword matching
            if is_diagnosis_intent(user_input):
                decision = {"action": "call_diagnose"}
            elif is_detection_intent(user_input):
                decision = {"action": "call_detect"}
            else:
                decision = {"action": "respond", "message": "您好！我是 AIOps 根因分析助手。"}

        logger.info(f"SUPERVISOR: Decision - {decision.get('action')}")

    except Exception as e:
        logger.error(f"SUPERVISOR: LLM error: {e}")
        # Conservative fallback: only route to diagnose on explicit keywords
        if is_diagnosis_intent(user_input):
            decision = {"action": "call_diagnose"}
        elif is_detection_intent(user_input):
            decision = {"action": "call_detect"}
        else:
            decision = {"action": "respond", "message": "您好！我是 AIOps 根因分析助手。请描述您需要分析的故障现象。"}

    # Execute decision
    action = decision.get("action", "respond")
    state["action"] = action

    if action == "call_diagnose":
        logger.info("SUPERVISOR: Routing to diagnose_agent (no parameter validation)")
        state["continue_conversation"] = True

    elif action == "call_detect":
        logger.info("SUPERVISOR: Routing to detect_agent (anomaly detection only)")
        state["continue_conversation"] = True

    elif action == "respond":
        response_message = decision.get("message", "")
        if not response_message:
            response_message = """您好！我是 AIOps 根因分析智能助手。

我可以帮您：
- 分析微服务异常事件的根因
- 分析指标数据，定位故障源头
- 生成故障传播图
- 使用 IAF-RCL 和 KE-FPC 算法进行综合诊断

请描述您的异常事件或提供数据文件，我将为您进行根因分析。

示例：
- "今天下午3点，我们的订单服务出现异常，请分析 data/metrics.csv"
- "帮我诊断这次故障，数据文件在 ./data/metrics.csv"
"""

        state["messages"] = [AIMessage(content=response_message)]
        state["continue_conversation"] = True

    return state


def build_supervisor_agent() -> StateGraph:
    """
    Build the supervisor agent subgraph.

    Simplified single-node graph that:
    1. Recognizes user intent
    2. Routes to diagnose_agent or responds directly
    3. NO parameter extraction or validation

    Returns:
        Compiled StateGraph for supervisor agent
    """
    logger.info("Building supervisor agent subgraph (intent-only, no parameter validation)")

    builder = StateGraph(SupervisorAgentState)

    # Add node
    builder.add_node("supervisor_llm", supervisor_llm_node)

    # Set entry point
    builder.set_entry_point("supervisor_llm")

    # End after supervisor - main graph will handle routing based on action
    builder.add_edge("supervisor_llm", END)

    # Compile
    graph = builder.compile()
    logger.info("Supervisor agent subgraph compiled")

    return graph


# Create global instance
supervisor_agent = build_supervisor_agent()