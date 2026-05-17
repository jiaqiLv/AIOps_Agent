# 微服务系统可观测数据因果发现系统

基于微服务时序指标数据的因果发现系统，利用PC/PCMCI算法构建因果图，支持专家知识约束和自动化评估。

## 项目特性

- **多算法支持**: 集成PC和PCMCI因果发现算法
- **智能预处理**: 自动处理缺失值、过滤常量和高相关性特征
- **知识驱动**: 基于正则表达式的指标分层和约束生成
- **模块化设计**: 清晰的模块分离，易于扩展和维护
- **独立评估**: 完整的评估工具，支持多种评估指标

## 项目结构

```
causal_discovery_system/
├── config/                    # 配置文件
│   ├── config.yaml           # 主配置文件
│   └── constraints.yaml      # 约束知识配置
├── data/                      # 数据目录
│   ├── raw/                  # 原始数据
│   ├── processed/            # 预处理后数据
│   └── output/               # 结果输出
├── src/                       # 源代码
│   ├── utils/                # 工具类
│   ├── preprocessing/        # 数据预处理
│   ├── knowledge/            # 知识管理
│   ├── algorithms/           # 算法引擎
│   └── main.py              # 主入口
├── tools/                     # 工具脚本
│   └── evaluate.py          # 评估工具
├── logs/                      # 日志文件
├── requirements.txt           # 依赖包
└── README.md                 # 项目说明
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository_url>
cd causal_discovery_system

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

将数据按照以下结构放置：

```
data/raw/{dataset_name}/{case_name}/
├── data.csv              # 原始指标数据
└── ground_truth.json     # 真实因果图（可选，用于评估）
```

数据格式示例：

**data.csv**:
```csv
timestamp,cpu_usage,memory_usage,latency,error_rate,revenue
2023-01-01 00:00:00,75.2,60.5,120.3,0.02,1000.5
2023-01-01 00:01:00,78.1,62.3,125.7,0.03,1050.2
...
```

**ground_truth.json**:
```json
{
  "nodes": ["cpu_usage", "memory_usage", "latency", "error_rate", "revenue"],
  "edges": [
    ["cpu_usage", "latency"],
    ["memory_usage", "latency"],
    ["latency", "error_rate"],
    ["error_rate", "revenue"]
  ]
}
```

### 3. 配置设置

编辑 `config/config.yaml` 设置运行参数：

```yaml
context:
  dataset_name: "your_dataset"
  case_name: "your_case"

data:
  auto_diff: true
  corr_threshold: 0.99

algorithm:
  name: "pc"  # 或 "pcmci"
  params:
    alpha: 0.05
    indep_test: "fisherz"
```

编辑 `config/constraints.yaml` 配置领域知识：

```yaml
metric_type_definitions:
  Resource:
    - ".*cpu.*"
    - ".*memory.*"
  QoS:
    - ".*latency.*"
    - ".*error_rate.*"
  Business:
    - ".*revenue.*"

level_definitions:
  Resource: 1    # 底层
  QoS: 2         # 中层
  Business: 3    # 高层
```

### 4. 运行因果发现

```bash
# 运行主流程
python src/main.py
```

运行结果将保存在 `data/output/{dataset}/{case}/` 目录下。

### 5. 评估结果（可选）

如果有真实因果图，可以使用评估工具：

```bash
python tools/evaluate.py \
  --pred_path data/output/your_dataset/your_case/final_graph.csv \
  --gt_path data/raw/your_dataset/your_case/ground_truth.json \
  --output_path evaluation_results.json
```

## 核心模块说明

### 数据预处理模块

- **缺失值处理**: 前向填充 + 零填充
- **常量过滤**: 移除方差为0的列
- **相关性过滤**: 移除高相关性特征（保留方差较大的）
- **智能差分**: 基于ADF检验的自动差分

### 知识管理模块

- **指标分类**: 基于正则表达式的自动分类
- **层级约束**: 高层级不能指向低层级
- **显式约束**: 支持用户指定的禁止/必需边

### 算法引擎

- **PC算法**: 基于约束的因果发现
- **PCMCI算法**: 适用于时间序列数据的因果发现
- **背景知识集成**: 支持先验约束

## 配置参数详解

### 主配置 (config.yaml)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `context.dataset_name` | 数据集名称 | 必填 |
| `context.case_name` | 案例名称 | 必填 |
| `data.auto_diff` | 是否启用智能差分 | true |
| `data.corr_threshold` | 相关性过滤阈值 | 0.99 |
| `algorithm.name` | 算法名称 (pc/pcmci) | pc |
| `algorithm.params` | 算法参数 | 见算法文档 |

### 约束配置 (constraints.yaml)

| 配置项 | 说明 |
|--------|------|
| `metric_type_definitions` | 指标类型正则定义 |
| `level_definitions` | 层级映射 |
| `explicit_forbidden` | 显式禁止边 |
| `explicit_required` | 显式必需边 |

## 算法参数

### PC算法参数

- `alpha`: 显著性水平 (默认: 0.05)
- `indep_test`: 独立性检验方法 (默认: fisherz)
- `stable`: 稳定版本 (默认: true)
- `uc_rule`: 无向边规则 (默认: 0)
- `uc_priority`: 无向边优先级 (默认: 2)

### PCMCI算法参数

- `tau_min`: 最小时间滞后 (默认: 1)
- `tau_max`: 最大时间滞后 (默认: 3)
- `pc_alpha`: PC阶段显著性水平 (默认: 0.05)
- `ci_test`: 条件独立性检验方法 (默认: par_corr)
- `max_cond_dim`: 最大条件集维度 (默认: 5)

## 评估指标

- **Precision**: 精确率
- **Recall**: 召回率
- **F1 Score**: F1分数
- **SHD**: 结构汉明距离
- **TP/FP/FN/TN**: 混淆矩阵元素

## 扩展开发

### 添加新算法

1. 在 `src/algorithms/` 目录下创建新算法类
2. 继承基础接口并实现 `run` 方法
3. 在 `AlgorithmFactory` 中注册新算法

### 自定义预处理

1. 修改 `src/preprocessing/data_processor.py`
2. 添加新的预处理步骤或修改现有逻辑

### 扩展约束类型

1. 修改 `src/knowledge/constraint_builder.py`
2. 添加新的约束生成逻辑

## 常见问题

### Q: 如何处理时间序列数据？
A: 使用PCMCI算法，它专门为时间序列设计，可以处理时间滞后效应。

### Q: 如何添加自定义的指标分类规则？
A: 在 `config/constraints.yaml` 中的 `metric_type_definitions` 添加新的正则表达式模式。

### Q: 预处理步骤可以跳过吗？
A: 可以通过修改配置文件中的 `data` 部分来控制预处理行为。

### Q: 如何调试算法运行过程？
A: 检查 `logs/` 目录下的日志文件，包含详细的运行信息。

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进项目。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 GitHub Issue
- 发送邮件至项目维护者