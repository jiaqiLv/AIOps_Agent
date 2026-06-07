# AIOps Agent

微服务故障根因分析助手。基于 LangGraph 的四 Agent Plan-and-Execute 架构，支持对话式异常检测与根因定位。

## 功能

- **多轮对话**：跨轮次保留 CSV 路径、故障时间、异常 KPI 等上下文
- **异常检测**：BLD Metric（ECOD 算法）识别异常指标
- **根因分析**：IAF-RCL（`rcd_tool`）+ KE-FPC（`pc_tool`）推理传播路径
- **报告生成**：汇总检测与诊断结果，输出自然语言报告
- **可视化**：故障传播拓扑图与 HTML 报告页面

## 架构

```
用户输入 → Supervisor（plan → execute → finalize）
                ├── Detection Agent   异常检测
                ├── Diagnose Agent    根因分析
                └── Report Agent      报告生成
```

Supervisor 按任务生成执行计划，调度三个子 Agent 完成检测、诊断与报告，最终返回分析结果。

## 快速开始

**环境要求：** Python 3.10+

```bash
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY

# 2. 一键安装依赖并启动 LangGraph 调试服务
./start_langgraph.sh

# 3. 或手动启动命令行交互模式
pip install -e . && pip install -r requirements.txt pyod
python -m app.main
```

### LangGraph 调试服务

`start_langgraph.sh` 启动后可访问：

| 地址 | 说明 |
|------|------|
| http://localhost:2024 | LangGraph API |
| http://localhost:8123 | LangGraph Studio 可视化界面 |
| `/topology/latest` | 最新故障传播图 |
| `/report/latest` | 最新分析报告 |

常用选项：

```bash
./start_langgraph.sh --skip-install   # 跳过依赖安装
./start_langgraph.sh --tunnel         # 开启公网隧道（无需端口映射）
```

### 命令行模式

```bash
# 交互式对话
python -m app.main

# 单次请求
python -m app.main --request "分析 ./data/sample_metrics.csv 文件的根因"
```

示例输入：

```
2026年1月5日5:48发生了故障，请检测异常指标
分析 ./data/RE1-OB/checkoutservice_cpu/1/data.csv 的根因
```

## 配置

`.env` 关键项：

```bash
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_key_here

# 拓扑图/报告链接前缀（外网访问时改为实际地址）
LANGGRAPH_PUBLIC_BASE_URL=http://192.168.199.5:32025
```

切换 LLM 提供商：设置 `LLM_PROVIDER` 为 `deepseek` / `openai` / `anthropic`，并填入对应 API Key。

## 项目结构

```
app/
├── agents/          # Supervisor + 子 Agent 图定义
├── middleware/      # LLM 重试、循环检测、会话上下文等
├── tools/           # CSV 读取、BLD Metric、RCD、PC、拓扑可视化
├── prompts/         # 各 Agent 的 Prompt 模板
├── config/          # agents.yaml、tools.yaml 配置
├── models/          # 状态定义与 LLM 工厂
└── main.py          # CLI 入口

data/                # 示例与测试数据集
start_langgraph.sh   # 一键启动脚本
langgraph.json       # LangGraph 服务配置
```

## 测试

```bash
pytest tests/
```

## 数据说明

| 路径 | 说明 |
|------|------|
| `data/sample_metrics.csv` | 简单示例 |
| `data/RE1-OB/*/data.csv` | RE1-OB 故障数据集 |
| `data/ZH_dataset/*/data.csv` | ZH 数据集 |
