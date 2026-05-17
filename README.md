# AIOps Diagnose Agent

AIOps 故障诊断 Agent 原型系统，基于 Supervisor-Subagent 架构。

## 功能特性

- **对话式交互**: 支持多轮对话，逐步收集诊断所需信息
- **Supervisor-Subagent 架构**: Supervisor 负责路由和对话管理，Diagnose Subagent 负责具体的诊断分析
- **CSV 文件读取**: 通过 LLM Tool Calling 自动调用 CSV 读取工具
- **DeepSeek LLM 集成**: 使用 DeepSeek 作为默认 LLM（支持切换其他模型）
- **LangGraph 工作流**: 使用 LangGraph 编排复杂的 Agent 工作流

## 架构说明

```
┌─────────────────────────────────────────────────────────┐
│                      Main Graph                          │
├─────────────────────────────────────────────────────────┤
│                                                           │
│   User Input                                             │
│       │                                                   │
│       ▼                                                   │
│   ┌─────────────┐                                       │
│   │ Supervisor  │ ◄─────────────┐                       │
│   │   Agent     │               │                       │
│   └──────┬──────┘               │                       │
│          │                      │                       │
│          │ 决定                  │                       │
│          ├──────────────────────┼─────────────────────┤│
│          │                      │                      ││
│      信息完整                  信息缺失                 ││
│          ▼                      ▼                      ││
│   ┌──────────────┐      ┌──────────────┐            ││
│   │ Diagnose     │      │ Ask for Info │            ││
│   │ Subagent     │      │   (Prompt    │            ││
│   │   (子图)      │      │    User)     │            ││
│   └──────────────┘      └──────────────┘            ││
│          │                                            ││
│          ▼                                            ││
│   返回诊断结果                                         ││
│                                                        ││
└─────────────────────────────────────────────────────────┘
```

### 组件说明

1. **Main Graph**: 顶层工作流，协调 Supervisor 和 Subagent
2. **Supervisor Agent**:
   - 分析用户输入
   - 提取诊断所需信息
   - 判断信息完整性
   - 决定下一步行动（直接回复 / 调用诊断子图 / 询问更多信息）
3. **Diagnose Subagent**:
   - 接收 Supervisor 收集的信息
   - 使用 LLM + Tool Calling 分析 CSV 数据
   - 生成诊断报告

## 快速开始

### 1. 安装依赖

```bash
pip install -e .
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件：
```bash
# DeepSeek API 配置
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# 可选：LangSmith 追踪
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=aiops-diagnose-agent
```

### 3. 运行应用

**交互式对话模式**（推荐）:
```bash
python -m app.main
```

**单次请求模式**:
```bash
python -m app.main --request "分析 checkoutservice 的 p99_latency 指标，数据文件 ./data/sample_metrics.csv"
```

### 4. 使用示例

```
============================================================
AIOps 故障诊断助手 v0.2.0
============================================================

我可以帮助您分析微服务故障指标。
请描述您遇到的问题，例如:
  "checkoutservice 服务的 p99_latency 指标飙升，
   数据文件在 ./data/sample_metrics.csv"

输入 'quit' 或 'exit' 退出
============================================================

您: checkoutservice 的 latency 指标有问题

助手: 为了进行故障诊断分析，我还需要以下信息:
1. CSV 数据文件的路径
2. 具体的指标名称（如 p99_latency, throughput 等）
3. 异常类型（飙升、下降、异常等）

您: 数据在 ./data/sample_metrics.csv，p99_latency 指标飙升

助手: 好的，我已收集到诊断所需的信息，正在为您进行分析...

=== 诊断分析报告 ===
...
```

## 项目结构

```
app/
├── graph/
│   ├── state.py              # 状态定义
│   ├── builder.py            # 主工作流构建器
│   ├── legacy_builder.py     # 旧版工作流（向后兼容）
│   ├── supervisor.py         # Supervisor Agent
│   └── nodes.py              # 传统节点（已废弃）
│
├── subagents/
│   ├── diagnose_subagent.py  # Diagnose Subagent（子图）
│   └── diagnose_agent.py     # Diagnose Agent（占位）
│
├── tools/
│   ├── langchain_csv_tool.py # LangChain 格式的 CSV 工具
│   ├── csv_reader_tool.py    # 传统 CSV 工具
│   ├── rcd_tool.py           # RCD 算法（占位）
│   └── pc_tool.py            # PC 算法（占位）
│
├── models/
│   ├── base.py               # LLM 基础接口
│   ├── deepseek.py           # DeepSeek 实现
│   └── model_factory.py      # 模型工厂
│
├── config/
│   ├── settings.py           # 应用配置
│   └── model_config.py       # 模型配置
│
├── prompts/
│   └── diagnose_prompt.py    # Prompt 模板
│
├── utils/                    # 工具函数
└── main.py                   # 应用入口
```

## 模型配置

项目支持多种 LLM 提供商，通过 `LLM_PROVIDER` 环境变量切换：

```bash
# DeepSeek（默认）
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_key

# OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key

# Anthropic
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_key
```

## LangGraph Studio

使用 LangGraph Studio 可视化调试：

```bash
langgraph dev
```

然后访问 http://localhost:8123 查看工作流图。

## 后续计划

- [ ] 实现 RCD 算法工具
- [ ] 实现 PC 算法工具
- [ ] 增强 Supervisor 的意图识别
- [ ] 支持多服务关联分析
- [ ] 添加更多诊断 Subagent
- [ ] 支持流式输出
