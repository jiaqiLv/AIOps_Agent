# 微服务系统可观测数据因果发现系统设计文档 (v2.0)

## 1. 项目概述

本项目旨在构建一个基于微服务时序指标数据的因果发现系统。系统核心目标是根据观测数据（CSV）和专家规则（YAML），利用因果发现算法（PC/PCMCI）构建因果图。

**核心变更点：**

- **数据目录**：采用 `raw/processed/output` 三层结构，并按 `数据集/案例` 嵌套隔离。
- **预处理**：指定 `ffill` + `fill(0)` 的缺失值处理策略。
- **知识层**：引入基于正则匹配的指标分层（Resource/QoS/Business）机制，自动生成“高层不能指向底层”的层级约束。
- **评估解耦**：评估模块独立为单独的脚本，不参与主流程。

## 2. 系统架构设计

### 2.1 目录结构规范

数据存储严格分层，支持多数据集和多案例管理。

```
causal_discovery_system/
├── config/
│   ├── config.yaml          # 全局运行参数（指定当前运行的数据集/案例）
│   └── constraints.yaml     # 领域先验知识（正则定义、层级定义、显式约束）
├── data/
│   ├── raw/                 # 原始数据层
│   │   └── {dataset_name}/
│   │       └── {case_name}/
│   │           ├── data.csv            # 原始指标数据
│   │           └── ground_truth.json   # 真实因果图标签
│   ├── processed/           # 预处理数据层
│   │   └── {dataset_name}/
│   │       └── {case_name}/
│   │           └── data_processed.csv  # 清洗/差分后的数据
│   └── output/              # 结果输出层
│       └── {dataset_name}/
│           └── {case_name}/
│               ├── intermediate/       # 中间结果(骨架图/相关性矩阵)
│               └── final_graph.csv     # 最终因果图(邻接矩阵)
├── logs/                    # 运行日志
├── src/
│   ├── __init__.py
│   ├── utils/               # 工具类(IO/Path/Logger)
│   ├── preprocessing/       # 预处理模块(含缺失值与相关性过滤)
│   ├── knowledge/           # 知识构建模块(解析YAML与生成约束)
│   ├── algorithms/          # 算法引擎(PC/PCMCI Wrapper)
│   └── main.py              # 核心主流程入口
├── tools/
│   └── evaluate.py          # [独立] 评估脚本
├── requirements.txt
└── README.md
```

### 2.2 数据流向

1. **Input**: `data/raw/{dataset}/{case}/data.csv`
2. **Process**: 读取 -> 填充/过滤/差分 -> 保存至 `data/processed/{dataset}/{case}/data_processed.csv`
3. **Knowledge**: 读取 `constraints.yaml` + 数据列名 -> 生成算法可用的 `PriorKnowledge` 对象。
4. **Solve**: 加载 Processed Data + Prior Knowledge -> 执行算法 -> 保存中间结果。
5. **Output**: 生成 `data/output/{dataset}/{case}/final_graph.csv`。

------

## 3. 详细模块设计

### 3.1 配置模块 (Config)

- **`config.yaml`**: 需包含当前任务上下文。

```yaml
context:
  dataset_name: "online_boutique"
  case_name: "case_01"

data:
  auto_diff: true
  corr_threshold: 0.99

algorithm:
  name: "pc"
  params: { ... }
```

### 3.2 数据预处理模块 (Preprocessing)

- **类名**: `DataProcessor`
- **输入**: 原始 DataFrame。
- **核心逻辑**:
  1. **缺失值处理 (Strict Order)**:
     - 第一步：`df.fillna(method='ffill')` (前向填充)。
     - 第二步：`df.fillna(0)` (剩余的NaN填充为0)。
  2. **常量过滤**: 删除方差为0的列。
  3. **高相关性过滤**: 计算Pearson相关系数，若 `corr > threshold`，保留方差较大的一列，删除另一列。
  4. **智能差分**: ADF检验 -> 若不平稳且线性相关 -> 差分。
- **输出**: 处理后的 DataFrame，并保存到 `processed` 目录。

### 3.3 知识管理模块 (Knowledge Manager)

- **类名**: `ConstraintBuilder`
- **输入**: `constraints.yaml` 配置对象，以及数据的 `column_names` 列表。
- **核心功能**:
  1. **指标归类 (Metric Classification)**:
     - 遍历所有列名，利用 YAML 中的正则表达式 (`metric_type_definitions`) 将每个指标归类为 `Resource` / `QoS` / `Business`。
     - 若某列无法匹配任何正则，标记为 `Unknown`。
  2. **层级映射 (Level Mapping)**:
     - 根据 `level_definitions` 将类型映射为 Level ID (1, 2, 3)。
  3. **生成层级约束 (Hierarchy Constraints)**:
     - **原则**: 因果流向为 Level 1 -> Level 2 -> Level 3。
     - **禁止**: 禁止从高 Level 指向低 Level 的边 (e.g., Forbidden: Business -> QoS, QoS -> Resource)。
     - 生成所有 `(High_Level_Node, Low_Level_Node)` 的禁止边列表。
  4. **合并显式约束 (Explicit Constraints)**:
     - 解析 `explicit_forbidden` 和 `explicit_required`。
     - 将“层级禁止边”与“显式禁止边”取并集。
- **输出**: 算法库所需的先验对象 (例如 causal-learn 的 `BackgroundKnowledge` 类或 forbidden/required 列表)。

### 3.4 算法引擎模块 (Algorithm Engine)

- **职责**: 接收 `pd.DataFrame` 和 `Constraints`，执行发现逻辑。
- **要求**:
  - 在保存中间结果时，需确保路径指向 `data/output/{dataset}/{case}/intermediate/`。
  - 日志中需打印：算法收敛步数、最终边数量、使用了多少条先验约束。

------

## 4. 独立评估模块设计 (Standalone Tool)

该模块完全独立，通过命令行运行，不属于 `main.py` 流程。

- **脚本**: `tools/evaluate.py`
- **输入参数**:
  - `--pred_path`: 预测图路径 (csv格式)
  - `--gt_path`: 真实图路径 (json格式)
  - `--output_path`: 评估结果保存路径 (json格式)
- **Ground Truth JSON 格式解析**:

```json
{
  "nodes": ["node_a", "node_b"],
  "edges": [ ["node_a", "node_b"] ] // a -> b
}
```

- **逻辑**:
  1. 读取预测 CSV (转换为邻接矩阵)。
  2. 读取 GT JSON (转换为邻接矩阵，需对齐节点顺序)。
  3. 计算 Precision, Recall, F1, SHD。
  4. 打印结果到控制台并保存文件。

------

## 5. 接口定义 (Python Pseudocode)

### 5.1 Knowledge Layer: Constraint Parser

```python
import re
import yaml

class KnowledgeParser:
    def __init__(self, constraints_config: dict):
        self.config = constraints_config
        # 加载正则预编译
        self.type_patterns = {
            k: [re.compile(p) for p in v] 
            for k, v in self.config['metric_type_definitions'].items()
        }
        self.levels = self.config['level_definitions']

    def parse(self, columns: list):
        """
        根据列名生成 Forbidden 和 Required 边列表
        """
        col_type_map = {}
        col_level_map = {}
        
        # 1. 识别类型和层级
        for col in columns:
            col_type = self._match_type(col)
            col_type_map[col] = col_type
            col_level_map[col] = self._get_level(col_type)

        forbidden_edges = set()
        required_edges = set()

        # 2. 生成层级约束 (High Level -> Low Level is Forbidden)
        for src in columns:
            for dst in columns:
                if src == dst: continue
                lvl_src = col_level_map.get(src)
                lvl_dst = col_level_map.get(dst)
                
                if lvl_src is not None and lvl_dst is not None:
                    # 如果 src 层级 > dst 层级 (例如 3->2, 2->1, 3->1)，则禁止 src->dst
                    # 注意：Level 1 是底层(Root Cause)，Level 3 是高层
                    if lvl_src > lvl_dst:
                        forbidden_edges.add((src, dst))

        # 3. 添加显式约束
        # ... 处理 explicit_forbidden / explicit_required ...
        
        return list(forbidden_edges), list(required_edges)
```

### 5.2 Main Workflow

```python
def main():
    # 1. Init Config & Logger
    cfg = ConfigManager("config/config.yaml")
    dataset = cfg.get("context.dataset_name")
    case = cfg.get("context.case_name")
    
    # Path Definitions
    raw_path = f"data/raw/{dataset}/{case}/data.csv"
    proc_path = f"data/processed/{dataset}/{case}/data_processed.csv"
    out_dir = f"data/output/{dataset}/{case}"
    
    # 2. Preprocess
    processor = DataPreprocessor(cfg.get("data"))
    df = pd.read_csv(raw_path)
    df_clean = processor.run(df)
    df_clean.to_csv(proc_path, index=False)
    
    # 3. Build Knowledge
    k_cfg = load_yaml("config/constraints.yaml")
    k_parser = KnowledgeParser(k_cfg)
    forbidden, required = k_parser.parse(df_clean.columns.tolist())
    
    # 4. Run Algorithm
    algo = AlgorithmFactory.get(cfg.get("algorithm.name"))
    result_graph = algo.run(df_clean, forbidden, required)
    
    # 5. Save Output
    save_graph(result_graph, out_dir + "/final_graph.csv")
```