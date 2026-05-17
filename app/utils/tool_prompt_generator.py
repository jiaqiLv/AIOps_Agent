"""Tool Description Generator for LLM Prompt Injection

This module generates tool descriptions and parameter specifications
to be injected into system prompts for LLM-based tool selection.
"""

from typing import List, Dict, Any
from langchain_core.tools import StructuredTool
from app.utils.logger import get_logger

logger = get_logger(__name__)


def generate_tools_prompt(tools: List[StructuredTool]) -> str:
    """Generate a formatted tools description for system prompt.

    Args:
        tools: List of LangChain StructuredTool instances

    Returns:
        Formatted string describing available tools
    """
    if not tools:
        return "# 可用工具\n\n当前没有可用的工具。"

    sections = ["# 可用工具\n"]
    sections.append("你可以使用以下工具来完成任务。请根据需要选择调用。\n")

    for i, tool in enumerate(tools, 1):
        sections.append(f"## {i}. {tool.name}\n")
        sections.append(f"**描述**: {tool.description}\n")

        # Extract parameter information from args_schema
        if tool.args_schema:
            sections.append("**参数**:\n")
            try:
                schema = tool.args_schema
                if hasattr(schema, 'model_fields'):
                    # Pydantic v2
                    fields = schema.model_fields
                elif hasattr(schema, '__fields__'):
                    # Pydantic v1
                    fields = schema.__fields__
                else:
                    fields = {}

                if fields:
                    for field_name, field_info in fields.items():
                        # Get field description
                        if hasattr(field_info, 'description'):
                            desc = field_info.description or ""
                        elif hasattr(field_info, 'field_description'):
                            desc = field_info.field_description or ""
                        else:
                            desc = ""

                        # Check if field is required
                        is_required = True
                        if hasattr(field_info, 'is_required'):
                            is_required = field_info.is_required()
                        elif hasattr(field_info, 'required'):
                            is_required = field_info.required
                        else:
                            # Check if there's a default value
                            if hasattr(field_info, 'default'):
                                is_required = field_info.default is ...  # Pydantic's Ellipsis for required

                        req_marker = "**必需**" if is_required else "可选"

                        # Get field type
                        if hasattr(field_info, 'annotation'):
                            type_str = str(field_info.annotation).replace('typing.', '').replace('ForwardRef(', '').replace("'", "")
                        else:
                            type_str = "str"

                        sections.append(f"- `{field_name}` ({type_str}, {req_marker}): {desc}\n")
                else:
                    sections.append("(无参数)\n")
            except Exception as e:
                logger.warning(f"Failed to extract schema for {tool.name}: {e}")
                sections.append("(参数信息提取失败)\n")

        sections.append("\n")

    # Add usage instructions
    sections.append("## 工具调用格式\n")
    sections.append("当你需要调用工具时，请按照以下格式：\n\n")
    sections.append("```\n")
    sections.append("调用: [工具名称]\n")
    sections.append("参数:\n")
    sections.append("  参数名1: 参数值1\n")
    sections.append("  参数名2: 参数值2\n")
    sections.append("```\n\n")

    sections.append("## 重要提示\n")
    sections.append("1. 你必须先分析用户的任务，提取出需要的参数\n")
    sections.append("2. 检查所有必需参数是否都已提供\n")
    sections.append("3. 如果参数缺失，请说明需要哪些参数\n")
    sections.append("4. 调用工具后，根据返回结果继续分析\n")

    return "".join(sections)


def generate_diagnose_agent_tools_prompt() -> str:
    """Generate the tools prompt for diagnose agent.

    This is a pre-generated version with custom optimized descriptions.
    """
    return """# 可用工具

你是根因分析Agent，必须使用以下工具来完成分析任务。

⚠️ **强制要求**: 你必须按顺序调用工具，不能跳过任何步骤！
1. csv_reader_tool（加载数据）
2. rcd_tool 或 pc_tool（分析根因）
3. **graph_visualization_tool（生成可视化图）← 第3步必须调用！**
4. 等待系统生成报告

## 1. csv_reader_tool - 读取CSV数据文件

**描述**: 加载微服务指标数据的CSV文件，返回数据的基本信息（列名、形状等）。这是第一个必须调用的工具。

**参数**:
- `data_path` (str, **必需**): CSV文件路径
  - 格式规则：从任务描述中提取日期，构造为 `data/ZH_dataset/{MMDD}/data.csv`
  - 示例：
    - "2026年1月5日" → `data/ZH_dataset/0105/data.csv`
    - "11月16日" → `data/ZH_dataset/1116/data.csv`
  - 注意：月日必须补零（1月→01，5日→05）

## 2. rcd_tool（IAF-RCL）- 快速根因推理

**描述**: 基于故障注入时间使用 IAF-RCL 算法识别根因指标，返回按优先级排序的候选根因列表。

**参数**:
- `inject_time` (float, **必需**): 故障注入时间
  - 从任务描述中提取时间，转换为Unix时间戳
  - 示例：
    - "2026年1月5日 5:48" → 需要转换为Unix时间戳
    - "1月5日 05:48" → 需要推断年份后转换
- `gamma` (int, 可选): IAF-RCL 算法参数，默认5
- `abnormal_kpi` (str, 可选): 异常指标名称，从任务中提取

## 3. pc_tool（KE-FPC）- 因果发现分析

**描述**: 使用 KE-FPC 算法进行因果发现，构建故障传播图，识别根因指标和因果关系。

**参数**:
- `alpha` (float, 可选): 显著性水平，默认0.05
- `abnormal_kpi` (str, 可选): 异常指标名称，从任务中提取

## 4. graph_visualization_tool - 故障传播图可视化

**描述**: 生成故障传播图的交互式可视化，用不同颜色区分KPI节点(红色)、根因节点(青色)和中间节点(浅青色)。

**参数**:
- `edges` (list, **必需**): 因果边列表，格式为 [[source, target], ...]
- `root_causes` (list, **必需**): 根因指标列表
- `abnormal_kpi` (str, 可选): 异常指标名称，用于高亮显示
- `output_format` (str, 可选): 输出格式，"html"(默认交互式) 或 "png"(静态图片)
- `output_dir` (str, 可选): 输出目录，默认 "outputs/graphs"

## 工作流程

你必须按照以下顺序执行：

### 第一步：解析任务并读取数据
从任务描述中提取：
1. **日期信息** → 构造 `data_path` 参数
2. **时间信息** → 转换为 `inject_time` 参数（Unix时间戳）
3. **异常指标** → 提取 `abnormal_kpi` 参数

然后调用 `csv_reader_tool` 读取数据。

### 第二步：执行根因分析
根据是否有 `inject_time` 决定：
- **有时间** → 调用 `rcd_tool` 进行快速推理
- **无时间** → 直接调用 `pc_tool` 进行因果发现
- **最佳实践** → 同时调用两个算法，交叉验证结果

### 第三步：可视化传播图（必须执行）
**重要**：在 KE-FPC 算法执行完成后，**你必须**调用 `graph_visualization_tool` 生成可视化！

这是分析的最后一步，不要跳过：
- 从 pc_tool 的结果中提取 `edges` 和 `root_causes`
- 调用 `graph_visualization_tool` 生成图
- 这将为用户提供直观的故障传播路径

调用示例：
```
调用: graph_visualization_tool
参数:
  edges: [["cpu_usage", "mem_usage"], ["mem_usage", "response_time"]]
  root_causes: ["cpu_usage", "disk_io"]
  abnormal_kpi: "response_time"
  output_format: html
```

### 第四步：生成分析报告
收集所有算法的结果和可视化图链接，生成结构化的根因分析报告。

## 参数提取规则

### 日期 → data_path
```
"2026年1月5日"     → data_path="data/ZH_dataset/0105/data.csv"
"11月16日"         → data_path="data/ZH_dataset/1116/data.csv"
"1月5日发生故障"   → data_path="data/ZH_dataset/0105/data.csv"
```
月份和日期都要补零：1→01, 5→05, 11→11

### 时间 → inject_time
需要转换为Unix时间戳（秒）：
- "2026-01-05 05:48:00" → 1736045280
- "1月5日 5:48" → 需要推断为2026-01-05 05:48:00 → 1736045280

### 异常指标 → abnormal_kpi
```
"full_request_duration_ms_new_10.104.128.205:9093飙升"
→ abnormal_kpi="full_request_duration_ms_new_10.104.128.205:9093"
```

## 调用示例

当收到任务："2026年1月5日5:48，微服务系统发生了故障，full_request_duration_ms_new_10.104.128.205:9093飙升。请结合指标数据，帮我分析下故障根因。"

你的分析过程应该是：

```
1. 解析任务：
   - 日期: 2026年1月5日 → MMDD=0105
   - 时间: 5:48 → 需要转换
   - 异常指标: full_request_duration_ms_new_10.104.128.205:9093

2. 调用csv_reader_tool:
   data_path="data/ZH_dataset/0105/data.csv"

3. 调用rcd_tool:
   inject_time=1736045280 (转换后的时间戳)
   abnormal_kpi="full_request_duration_ms_new_10.104.128.205:9093"

4. 调用pc_tool:
4. 调用pc_tool:
   abnormal_kpi="full_request_duration_ms_new_10.104.128.205:9093"

5. 调用graph_visualization_tool:
   调用: graph_visualization_tool
   参数:
     edges: [["cpu_usage", "mem_usage"], ["mem_usage", "response_time"]]
     root_causes: ["cpu_usage", "disk_io"]
     abnormal_kpi: "full_request_duration_ms_new_10.104.128.205:9093"
     output_format: html

6. 综合结果生成报告
```

6. 综合结果生成报告
```

## 错误处理

- 如果CSV文件读取失败，说明路径可能错误，请根据日期重新构造
- 如果时间转换失败，请尝试常见格式或说明需要完整时间
- 如果算法执行失败，尝试其他算法或说明失败原因
- **绝不能**因为参数缺失就直接回复用户，必须先尝试提取或说明需求
"""


def inject_tools_into_prompt(base_prompt: str, tools: List[StructuredTool]) -> str:
    """Inject tools description into system prompt.

    Args:
        base_prompt: Base system prompt
        tools: List of tools to describe

    Returns:
        Enhanced prompt with tools description
    """
    tools_prompt = generate_tools_prompt(tools)
    return f"{base_prompt}\n\n{tools_prompt}"
