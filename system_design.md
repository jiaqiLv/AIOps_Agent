# AIOps Agent 系统设计文档

> 版本: 3.0 | 日期: 2026-05-29
> 用途: 作为代码重构的参考蓝图。本文档描述目标架构，实际代码按本文档生成。

---

## 1. 系统概述

AIOps Agent 是一个对话式微服务异常根因分析系统。用户以自然语言描述异常事件或提供指标数据，系统自动完成异常检测与根因定位，输出结构化的分析报告。

### 1.1 核心能力

| 能力 | 说明 |
|------|------|
| 异常检测 | 3-Sigma 统计方法，识别偏离基线的异常指标 |
| 根因定位 | IAF-RCL (排序算法) + KE-FPC (因果发现算法) |
| 传播拓扑 | 基于因果图生成故障传播路径可视化 |
| 报告合成 | LLM 整合检测结果与算法输出，生成统一分析报告 |

### 1.2 运行模式

| 模式 | 启动方式 | 用途 |
|------|----------|------|
| CLI 交互 | `python -m app.main` | 命令行对话 |
| CLI 单次 | `python -m app.main --request "..."` | 单次请求 |
| LangGraph Studio | `langgraph dev` | 可视化调试 |

---

## 2. 系统架构

### 2.1 三 Agent 架构总览（Plan-and-Execute）

```
┌─────────────────────────────────────────────────────────────────┐
│  Main Graph                                                      │
│  START → supervisor_node → END                                   │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  Supervisor Agent (Plan-and-Execute)                      │    │
│  │                                                            │    │
│  │  planner ──empty──→ direct_reply → END                    │    │
│  │    │                                                      │    │
│  │    └──has steps──→ executor ──→ executor (loop) ──→       │    │
│  │                                        reporter → END     │    │
│  │                                                           │    │
│  │  Registry: SubAgentAdapter (build_input / extract_result) │    │
│  │  Memory: step_results[step_id] → dict                     │    │
│  │                                                            │    │
│  │  ┌─────────────────────┐  ┌─────────────────────────┐     │    │
│  │  │ Detection Adapter   │  │ Diagnose Adapter        │     │    │
│  │  │ ────────────────── │  │ ─────────────────────── │     │    │
│  │  │ csv_reader_tool     │  │ csv_reader_tool         │     │    │
│  │  │ three_sigma_tool    │  │ rcd_tool (IAF-RCL)      │     │    │
│  │  │                     │  │ pc_tool (KE-FPC)         │     │    │
│  │  │                     │  │ graph_visualization_tool │     │    │
│  │  └─────────────────────┘  └─────────────────────────┘     │    │
│  └──────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Agent 职责划分

| Agent | 职责 | 模型温度 | 最大迭代 | 输出 |
|-------|------|---------|---------|------|
| **Supervisor** | 任务规划 + 步骤调度 + 报告合成 | 规划: 0, 合成: 0.3 | N/A (plan-driven) | 最终报告文本 |
| **Detection** | 数据加载 + 3-Sigma 异常检测 | 0 | 5 | 异常摘要 + 参数 |
| **Diagnose** | 数据加载 + IAF-RCL/KE-FPC 分析 | 0 | 10 | 结构化算法结果 |

### 2.3 数据流

```
用户输入
  │
  ▼
MainState (user_input, messages)
  │
  ▼ 转换
PlanExecuteState
  │
  ▼ planner (LLM, temp=0)
  ├─→ 空计划 → direct_reply (对话回复)
  │
  ├─→ [detection step] → DetectionAdapter → DetectionAgent → step_results[1]
  │
  ├─→ [diagnose step] → DiagnoseAdapter(step_results[1]) → DiagnoseAgent → step_results[2]
  │
  ▼ reporter (模板渲染 + LLM, temp=0.3)
final_response (统一报告)
  │
  ▼ 转换
MainState (messages, diagnose_result)
```

---

## 3. 模块详细设计

### 3.1 目录结构（目标）

```
app/
├── main.py                          # CLI 入口
├── http_app.py                      # HTTP 路由 (LangGraph Studio)
│
├── agents/                          # Agent 定义
│   ├── __init__.py                  # 模块导出
│   ├── main_graph.py                # 主图: START → supervisor → END
│   ├── supervisor_plan_execute.py   # Supervisor Plan-Execute (planner → executor → reporter)
│   ├── subgraph_registry.py         # SubAgentAdapter 接口 + 注册表
│   ├── detection_agent.py           # Detection ReAct
│   ├── diagnose_agent.py            # Diagnose ReAct
│   ├── nodes/                       # 通用 ReAct 节点
│   │   ├── __init__.py
│   │   └── react_nodes.py           # model / extract / final / routing
│   └── tools/                       # Agent 工具
│       ├── __init__.py
│       └── (原子工具在此)
│
├── models/                          # 状态 & 数据模型
│   ├── __init__.py
│   ├── plan_execute_state.py        # PlanExecuteState + PlanStep
│   ├── detection_agent_state.py     # DetectionAgentState
│   ├── react_agent_state.py         # ReactAgentState (Diagnose)
│   ├── schemas.py                   # 工具 Pydantic schemas
│   ├── base.py                      # LLM 基类
│   ├── deepseek.py                  # DeepSeek 实现
│   └── model_factory.py            # LLM 工厂
│
├── tools/                           # 工具实现
│   ├── csv_reader_tool.py           # CSV 数据加载
│   ├── three_sigma.py               # 3-Sigma 异常检测
│   ├── rcd_wrapper.py               # IAF-RCL 算法包装
│   ├── pc_wrapper.py                # KE-FPC 算法包装
│   ├── graph_visualization_tool.py  # 拓扑可视化
│   ├── human_tool.py                # 用户交互
│   ├── langchain_tool_adapters.py   # 工具适配器
│   ├── tool_registry.py             # 工具注册表
│   └── rcd/                         # IAF-RCL 算法实现
│       ├── rcd.py
│       └── time_series.py
│
├── config/                          # 配置
│   ├── model_config.py              # LLM 配置
│   ├── settings.py                  # 环境变量
│   └── agents.yaml                  # Agent 配置声明
│
├── prompts/                         # Prompt 模板
│   ├── supervisor_planner.md        # Supervisor Planner 系统提示
│   ├── supervisor_synthesis.md      # 报告合成模板
│   ├── detection_system.md          # Detection 系统提示
│   └── diagnose_system.md           # Diagnose 系统提示
│
├── utils/                           # 工具函数
│   ├── logger.py                    # 日志
│   ├── llm_logger.py                # LLM 对话日志
│   ├── path_resolver.py             # 路径解析
│   ├── prompt_loader.py             # Prompt 加载
│   ├── prompt_template.py           # 模板渲染
│   ├── lazy_graph.py                # 延迟图加载
│   ├── json_utils.py                # JSON 工具
│   ├── propagation_paths.py         # 传播路径提取
│   ├── topology_chat.py             # Studio 拓扑格式化
│   ├── topology_embed.py            # 嵌入式拓扑
│   └── validation.py                # 输入校验
│
├── skills/                          # 技能模块
│   ├── data_loading_skill.py
│   └── result_formatting_skill.py
│
└── static/                          # 静态资源
    └── topology/
```

### 3.2 State 设计

#### 3.2.1 State 层级与职责

**原则: 每个 State 只定义自己需要的字段，不混入其他 Agent 的字段。**

```
MainState (主图)
  ├── user_input, messages, ui
  ├── csv_file_path, inject_time, abnormal_kpi  (参数传播)
  ├── diagnose_result                            (最终结果)
  └── final_response, continue_conversation

PlanExecuteState (Supervisor Plan-Execute)
  ├── messages, user_input, task_description
  ├── plan: List[PlanStep]                        (执行计划)
  ├── current_step_index, plan_reasoning          (计划状态)
  ├── step_results: Dict[int, Dict]              (通用 memory)
  └── final_response, continue_conversation

PlanStep (计划步骤)
  ├── step_id, name, agent                        ("detection" | "diagnose")
  ├── input: Dict[str, Any]                       (步骤输入)
  └── status, error                               (执行状态)

DetectionAgentState (Detection ReAct 循环)
  ├── messages, task_description
  ├── iteration_count, max_iterations, tool_errors, tool_results
  ├── csv_file_path, inject_time, abnormal_kpi    (输入参数 + 输出)
  └── three_sigma_result, final_response

DiagnoseAgentState (Diagnose ReAct 循环)
  ├── messages, task_description
  ├── iteration_count, max_iterations, tool_errors, tool_results
  ├── csv_file_path, inject_time, abnormal_kpi    (输入参数)
  ├── rcd_result, pc_result                       (算法结果)
  ├── graph_visualizations                        (可视化)
  └── final_response, integrated_result           (结构化输出)
```

#### 3.2.2 State 转换规则

| 转换点 | 源 State | 目标 State | 方向 | 说明 |
|--------|----------|-----------|------|------|
| supervisor_node | MainState | PlanExecuteState | 正向 | 映射 user_input → task_description，初始化空计划 |
| supervisor_node | PlanExecuteState | MainState | 反向 | 提取 final_response、step_results → diagnose_result、new AI messages |
| DetectionAdapter | PlanStep.input + step_results | DetectionAgentState | 正向 | task_description + 可选参数 |
| DetectionAdapter | DetectionAgentState | step_results[step_id] | 反向 | summary + csv_file_path + inject_time + abnormal_kpi |
| DiagnoseAdapter | PlanStep.input + step_results | DiagnoseAgentState | 正向 | task_description + 可选参数 + from_step 引用 |
| DiagnoseAdapter | DiagnoseAgentState | step_results[step_id] | 反向 | rcd_result + pc_result + graph_visualizations |

### 3.3 Plan-Execute 设计（Supervisor）

#### 3.3.1 Plan-Execute 模式

Supervisor 使用 Plan-Execute 模式，Sub-agent 使用 ReAct 模式：

```
Supervisor (Plan-Execute):
                    ┌──────────┐
                    │  START   │
                    └────┬─────┘
                         ▼
                    ┌──────────┐
                    │ planner  │ (LLM, temp=0)
                    └────┬─────┘
                         │
              ┌──────────┴──────────┐
              │                     │
         empty plan           has steps
              │                     │
              ▼                     ▼
       ┌──────────┐          ┌──────────┐
       │direct    │          │ executor │◄──────────┐
       │_reply    │          └────┬─────┘           │
       └────┬─────┘               │                 │
            │              step executed             │
            │                     │            more steps
            │                     ▼                 │
            │              ┌──────────┐              │
            │              │ reporter │              │
            │              │(LLM temp │              │
            │              │ =0.3)    │              │
            │              └────┬─────┘              │
            │                   │                    │
            ▼                   ▼                    │
           END                 END                   │
                                │                    │
                                └────────────────────┘

Sub-agents (ReAct):
                    ┌──────────┐
                    │  START   │
                    └────┬─────┘
                         ▼
                    ┌──────────┐
              ┌─────│  model   │◄────────────┐
              │     └────┬─────┘              │
              │          │                     │
         tool_calls   no_calls              tool_calls
              │          │               (loop back)
              ▼          ▼                     │
       ┌──────────┐  ┌─────────────┐           │
       │  tools   │  │ final       │           │
       └────┬─────┘  └──────┬──────┘           │
            │               │                  │
            ▼               │                  │
       ┌──────────┐         │                  │
       │ extract  │─────────┘                  │
       │ _results │────────────────────────────┘
       └──────────┘
```

```
                    ┌──────────┐
                    │  START   │
                    └────┬─────┘
                         ▼
                    ┌──────────┐
              ┌─────│  model   │◄────────────┐
              │     └────┬─────┘              │
              │          │                     │
         tool_calls   no_calls              tool_calls
              │          │               (loop back)
              ▼          ▼                     │
       ┌──────────┐  ┌─────────────┐           │
       │  tools   │  │ 终止节点     │           │
       └────┬─────┘  │(final/      │           │
            │        │ synthesis)  │           │
            ▼        └──────┬──────┘           │
       ┌──────────┐         │                  │
       │ extract  │─────────┘ (no more tools)  │
       │ _results │────────────────────────────┘
       └──────────┘     (more tools pending)
```

#### 3.3.2 Supervisor 节点定义

| 节点 | 函数 | 职责 |
|------|------|------|
| **planner** | `planner_node(state)` | 分析用户请求，调用 LLM 生成执行计划 (JSON) |
| **executor** | `executor_node(state)` | 按 plan 步骤调用 subgraph (via adapter)，结果存入 step_results |
| **direct_reply** | `direct_reply_node(state)` | 对话式回复 (无子 Agent) |
| **reporter** | `reporter_node(state)` | 模板渲染 + LLM 合成报告 (temp=0.3) |

#### 3.3.3 Sub-agent 节点定义

| 节点 | 工厂函数 | 职责 |
|------|---------|------|
| **model** | `create_model_node(llm, system_prompt, first_iteration_instruction)` | 组装 system + history + instruction，调用 LLM，返回 AIMessage |
| **tools** | `ToolNode(tools)` (LangGraph 内置) | 执行 tool_calls，返回 ToolMessages |
| **extract_results** | `extract_results_node(state)` | 解析 ToolMessages，更新 State 字段，检测中断 |
| **final** | Agent 各自定义 | 输出最终结果 |

#### 3.3.4 路由函数

| 路由点 | 函数 | 路径选项 |
|--------|------|---------|
| planner → | `route_after_planner` | "executor" / "direct_reply" |
| executor → | `route_after_executor` | "executor" (loop) / "reporter" |
| model → (sub-agents) | `route_after_model` | "tools" / "final" |
| extract → (sub-agents) | `route_after_extract` | "model" / "interrupt" / "final" |

#### 3.3.5 Agent 差异矩阵

| 特性 | Supervisor (Plan-Execute) | Detection (ReAct) | Diagnose (ReAct) |
|------|---------------------------|-------------------|-------------------|
| 编排模式 | planner → executor → reporter | model → tools → extract → model (loop) | model → tools → extract → model (loop) |
| 子 Agent 调用 | via SubAgentAdapter + subgraph.invoke() | N/A | N/A |
| 结果存储 | step_results[step_id] | state 字段 | state 字段 + integrated_result |
| 报告生成 | reporter node (LLM temp=0.3) | final node (无 LLM) | final node (无 LLM, 打包 JSON) |

### 3.4 工具设计

#### 3.4.1 工具分层

```
工具层
├── 原子工具 (app/tools/)
│   ├── csv_reader_tool       # CSV 读取、缓存、列名规范化
│   ├── three_sigma_tool      # 3-Sigma 统计检测
│   ├── rcd_tool              # IAF-RCL 根因排序
│   ├── pc_tool               # KE-FPC 因果发现
│   ├── graph_visualization_tool  # HTML/PNG 拓扑图
│   └── ask_user              # 交互式中断
│
├── 工具适配器 (app/tools/langchain_tool_adapters.py)
│   └── 将原子工具包装为 LangChain StructuredTool
│       (Pydantic schema → 参数校验 → JSON 返回)
│
└── 子 Agent 适配器 (app/agents/subgraph_registry.py)
    ├── SubAgentAdapter       # 统一接口: build_input / extract_result
    ├── DetectionAdapter      # detection subgraph 输入/输出映射
    └── DiagnoseAdapter       # diagnose subgraph 输入/输出映射 (含 from_step)
```

#### 3.4.2 工具接口规范

所有工具遵循统一接口：

```python
# 输入: Pydantic BaseModel
class XxxInput(BaseModel):
    required_param: str = Field(..., description="...")
    optional_param: Optional[str] = Field(default=None, description="...")

# 输出: JSON 字符串，必须包含 success 字段
{
    "success": true,
    "data": {...},          # 成功时
    "error": "message"      # 失败时
}
```

#### 3.4.3 工具分配

| Agent | 工具 |
|-------|------|
| **Supervisor** | SubAgentAdapter (detection, diagnose via subgraph_registry) |
| **Detection** | csv_reader_tool, three_sigma_tool |
| **Diagnose** | csv_reader_tool, rcd_tool, pc_tool, graph_visualization_tool |

### 3.5 模型配置

| Agent | 用途 | 温度 | 最大 Token |
|-------|------|------|-----------|
| Supervisor | 决策 (bind_tools) | 0 | 默认 |
| Supervisor | 报告合成 | 0.3 | 默认 |
| Detection | 工具调用 | 0 | 默认 |
| Diagnose | 工具调用 | 0 | 默认 |

所有 LLM 实例通过 `model_factory` 创建，支持 provider 切换 (DeepSeek / OpenAI / Anthropic)。

### 3.6 Prompt 管理

#### 3.6.1 Prompt 文件清单

| 文件 | 用途 | 使用者 |
|------|------|--------|
| `supervisor_planner.md` | Supervisor 任务规划指导 (子 Agent 描述、决策规则、JSON 输出格式) | Supervisor planner_node |
| `supervisor_synthesis.md` | 报告合成模板，含 `{{TASK_DESCRIPTION}}`, `{{DETECTION_SUMMARY}}`, `{{DIAGNOSE_RESULT}}` | Supervisor reporter_node |
| `detection_system.md` | Detection 工具调用指导 | Detection model_node |
| `diagnose_system.md` | Diagnose 工具调用指导 | Diagnose model_node |

#### 3.6.2 模板渲染规则

- Prompt 文件使用 Markdown 格式
- 变量占位符: `{{VARIABLE_NAME}}`
- 渲染通过 `prompt_template.render_template(path, variables)` 完成
- 仅 `supervisor_synthesis.md` 使用模板变量，其他为纯文本

### 3.7 错误处理

```
错误处理原则:
1. 工具错误不中断执行 — 记录到 state["tool_errors"]
2. LLM 可以根据错误信息决定重试、跳过或请求用户输入
3. 最终报告包含所有成功结果和错误信息
4. 子 Agent 异常由 subagent_tools 捕获，返回 {success: false, error: "..."}
```

---

## 4. 关键流程

### 4.1 完整分析流程（典型路径）

```
 1. 用户输入: "分析 data/ZH_dataset/0105/data.csv 的根因"
 2. Main Graph → supervisor_node
 3. supervisor_node 构建 PlanExecuteState, 调用 plan_execute_agent.invoke()
 4. planner_node:
    a. 加载 supervisor_planner.md 提示词
    b. 调用 LLM (temp=0) 生成执行计划 JSON
    c. 解析 JSON → List[PlanStep]
    d. plan = [step1(detection), step2(diagnose)]
 5. route_after_planner → "executor" (有步骤)
 6. executor_node (step 1: detection):
    a. DetectionAdapter.build_input(step1.input, step_results={})
    b. detection_agent.invoke(input_state)
       - Detection ReAct: csv_reader_tool → three_sigma_tool → final
    c. DetectionAdapter.extract_result(output) → result1
    d. step_results[1] = result1
    e. current_step_index = 1
 7. route_after_executor → "executor" (更多步骤)
 8. executor_node (step 2: diagnose):
    a. DiagnoseAdapter.build_input(step2.input, step_results={1: result1})
       - step2.input["from_step"]=1 → 自动获取 detection 摘要和参数
    b. diagnose_agent.invoke(input_state)
       - Diagnose ReAct: csv_reader_tool → rcd_tool → pc_tool → graph_viz → final
    c. DiagnoseAdapter.extract_result(output) → result2
    d. step_results[2] = result2
    e. current_step_index = 2
 9. route_after_executor → "reporter" (计划完成)
10. reporter_node:
    a. 从 plan 识别 detection 和 diagnose 步骤
    b. format_detection_summary(step_results[1]) → 检测摘要文本
    c. format_diagnose_summary(step_results[2]) → 诊断摘要文本
    d. render_template("supervisor_synthesis.md", {...}) → 完整 prompt
    e. LLM (temp=0.3) 生成最终报告
    f. 附带拓扑可视化 (build_final_report_message)
11. plan_execute_agent 返回 PlanExecuteState
12. supervisor_node: 映射回 MainState, 从 step_results 提取 diagnose_result
13. 返回用户
```

### 4.2 直接对话流程（无子 Agent）

```
1. 用户输入: "你好，你能做什么？"
2. planner_node: LLM 生成空计划 (steps=[])
3. route_after_planner → "direct_reply"
4. direct_reply_node: 返回问候/直接文本回复
5. 返回用户
```

---

## 5. 配置与部署

### 5.1 环境变量 (.env)

```bash
# LLM 配置
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
LLM_TEMPERATURE=0.7

# 调试
LANGGRAPH_DEBUG=true
```

### 5.2 LangGraph Studio 配置 (langgraph.json)

```json
{
  "dependencies": ["."],
  "http": {
    "app": "./app/http_app.py:app"
  },
  "graphs": {
    "main": "./app/agents/main_graph.py:main_graph",
    "supervisor": "./app/agents/supervisor_plan_execute.py:build_plan_execute_agent",
    "diagnose": "./app/agents/diagnose_agent.py:build_diagnose_agent"
  },
  "env": ".env"
}
```

> 注意: `supervisor` 和 `diagnose` 引用 `build_*` 工厂函数，因为 LangGraph 运行时不识别 `LazyGraph` 代理。`main_graph` 直接引用编译后的图实例。

---

## 6. 当前问题与重构建议

### 6.1 已解决（v3.0 Plan-Execute 重构）

以下问题已通过 Plan-Execute 重构解决：

- **Supervisor ReAct 过度设计** → 改为 Plan-Execute 模式，LLM 只做规划不做决策循环
- **subagent_tools.py 状态映射 shim** → SubAgentAdapter 模式，通用接口 build_input / extract_result
- **detection_result/diagnose_result 独立字段** → 通用 step_results[step_id] memory
- **硬编码 executor 逻辑** → Adapter 注册表模式
- **路由函数三套** → Plan-Execute 只有 planner/executor/reporter 三路由

### 6.2 State 管理问题

**问题 1: ReactAgentState 职责过重**

`ReactAgentState` 同时服务于 Diagnose Agent，但包含了大量 legacy 字段（`csv_headers`, `gamma`, `alpha`, `three_sigma_result`），这些字段对 Diagnose Agent 无意义。

> **建议**: 重命名为 `DiagnoseAgentState`，删除 Diagnose Agent 不使用的字段（`csv_headers`, `gamma`, `alpha`, `three_sigma_result`）。

**问题 2: ReactAgentState 自定义 add_messages**

`ReactAgentState` 定义了本地 `add_messages` 函数而非使用 `langgraph.graph.add_messages`。

> **建议**: 统一使用 `from langgraph.graph import add_messages`。

### 6.3 extract_results_node 问题

**问题 3: 硬编码工具名 + 巨型函数**

`extract_results_node` 使用 `if/elif` 硬编码了工具名的解析逻辑。每新增一个工具都需要修改此函数。

> **建议**: 采用工具结果提取器注册机制:
> ```python
> EXTRACTORS = {
>     "csv_reader_tool": extract_csv_result,
>     "three_sigma_tool": extract_three_sigma_result,
>     "rcd_tool": extract_rcd_result,
>     ...
> }
> def extract_results_node(state):
>     for tool_msg in tool_messages:
>         extractor = EXTRACTORS.get(tool_msg.name)
>         if extractor:
>             extractor(state, result)
> ```

### 6.4 LazyGraph 问题

**问题 4: LazyGraph 与 LangGraph Studio 不兼容**

`LazyGraph` 代理类不被 LangGraph 运行时识别为合法的 Graph 或工厂函数。`langgraph.json` 已改为引用 `build_*` 函数，但程序化调用仍通过 `LazyGraph`。

> **建议**: 保留 `LazyGraph` 用于程序化调用（避免 import 时初始化 LLM），但确保 `langgraph.json` 始终引用 `build_*` 工厂函数。

### 6.5 配置碎片化

**问题 5: agents.yaml 声明但未完全使用**

`agents.yaml` 定义了完整的 Agent 配置（state_schema, tools, prompts），但实际代码中各 Agent 直接硬编码构建，未从 YAML 读取。

> **建议**: 要么让代码真正读取 agents.yaml（实现配置驱动），要么删除 agents.yaml 中冗余的声明，只保留有实际作用的配置。

### 6.6 模块职责模糊

**问题 6: skills/ 目录未集成**

`app/skills/` 下的 `data_loading_skill.py` 和 `result_formatting_skill.py` 未被任何代码引用。

> **建议**: 如果计划使用，集成到 Agent 流程中；如果废弃，删除以避免混淆。

**问题 7: config/ 中多个未使用文件**

`tools.yaml`, `workflows.yaml`, `params.yaml`, `tool_config.py`, `algorithm_names.py` 存在但未在代码中实际读取。

> **建议**: 清理未使用的配置文件，或将代码改为配置驱动。

---

## 7. 重构优先级

| 优先级 | 任务 | 影响范围 | 风险 |
|--------|------|---------|------|
| P0 | State 重命名与清理 (问题 1-2) | models/ | 低 |
| P0 | 删除未使用模块 (问题 6-7) | skills/, config/ | 低 |
| P1 | extract_results 解耦 (问题 3) | react_nodes.py | 中 |
| P2 | 配置统一 (问题 5) | config/, agents/ | 中 |
| P3 | LazyGraph 文档化 (问题 4) | lazy_graph.py | 低 |

---

## 8. 测试策略

| 测试类型 | 文件 | 覆盖范围 |
|---------|------|---------|
| 单元测试 | test_csv_tool.py | CSV 读取、缓存 |
| 单元测试 | test_models.py | LLM 工厂、provider 切换 |
| 集成测试 | test_integration.py | 三 Agent 端到端流程 |
| 集成测试 | test_new_workflow.py | Supervisor-Agent 工作流 |
| 入口测试 | test_main.py | CLI 启动 |

---

## 附录 A: 算法说明

### IAF-RCL (rcd_wrapper.py)

- **全称**: Iterative Anomaly Framework - Root Cause Localization
- **输入**: inject_time (必需), gamma (默认 5)
- **输出**: 排序后的根因指标列表
- **实现**: `app/tools/rcd/rcd.py`

### KE-FPC (pc_wrapper.py)

- **全称**: Knowledge-Enhanced Fault Propagation Construction
- **输入**: alpha (默认 0.05)
- **输出**: 根因指标 + 因果图边列表
- **后备**: 当 causal-learn 不可用时，降级为相关性分析

### 3-Sigma (three_sigma.py)

- **输入**: 数据窗口参数
- **输出**: 异常指标列表、注入时间、异常 KPI
- **方法**: 基于均值 +/- 3sigma 的统计异常检测
