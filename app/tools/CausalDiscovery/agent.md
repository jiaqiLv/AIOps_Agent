## 1. 顶层：控制中心 (Supervisor Agent)

- **大脑 (Brain)**：
  - 大语言模型：Deepseek-V3/R1 系列 LLM 。
  - 系统提示词：你是一个 AIOps 专家，...
- **记忆 (Memory)**：
  - 短期记忆（buffer）:
  - 长期知识库(Vector DB)：Sop 文档、历史案例等。
- **工具（Tools)：**
  - Observability Agent
  - Detection Agent
  - Diagnose Agent
  - ...
- **动态工作流（遵循ReAct范式）**：
  - **感知**：解析用户查询 。
  - **规划**：生成任务图 (DAG) 。
  - **执行**：向子智能体下达指令 。
  - **反思**：检查一致性（若证据矛盾则循环回“执行”）。
  - **输出**：生成最终诊断报告（包含根因、传播路径、解释叙述） 。