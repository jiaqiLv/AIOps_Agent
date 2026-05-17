# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **microservice causal discovery system** that builds causal graphs from time series observable data. It uses PC/PCMCI algorithms with domain knowledge constraints, followed by post-processing for edge orientation and root cause analysis.

## Running the System

```bash
# RADICE dataset (single sample)
python run.py --config config/config.yaml

# RADICE dataset (batch processing)
python batch.py --config config/config.yaml --dataset N15

# ZH dataset
python run_zh.py --config config/zh_config.yaml --sample 0105

# Or use the unified script:
./run.sh              # Run RADICE single sample
./run.sh batch        # Run RADICE batch processing
./run.sh zh           # Run ZH dataset
```

## Configuration System

**All runtime parameters are centralized in YAML files:**

- `config/config.yaml` - RADICE dataset configuration
- `config/zh_config.yaml` - ZH dataset configuration
- `config/constraints.yaml` - Domain knowledge constraints

Before running, update `context.case_name` (for RADICE) or `context.sample_name` (for ZH) in the respective config file.

**Data paths follow a strict three-tier structure:**
```
data/
├── raw/{dataset}/{case}/data.csv          # Input data
├── processed/{dataset}/{case}/data_processed.csv  # Preprocessed output
└── output/{dataset}/{case}/
    ├── final_graph.csv                    # Main causal graph result
    ├── intermediate/                      # Algorithm intermediates
    └── rca_results.csv                    # Root cause analysis (optional)
```

## Architecture

### Main Pipeline (`run.py`, `run_zh.py`, `batch.py`)

1. **Load & Filter**: Load CSV/RADICE data, apply time window (if configured)
2. **Preprocess**: Missing values → Wavelet denoising → Constant removal → Metric clustering → Correlation filtering → Optional differencing
3. **Build Knowledge**: Parse constraints.yaml to generate forbidden/required edges via regex-based metric classification and hierarchy rules
4. **Run Algorithm**: PC/PCMCI with custom CI test that filters conditioning sets by level (prevents conditioning on downstream variables)
5. **Post-process**: Orient undirected edges using background knowledge → time lag → conditional entropy
6. **RCA**: Root cause analysis using PageRank/random walk on final graph
7. **Evaluation**: Graph metrics (F1, SHD) and RCA accuracy (for RADICE)

### Key Modules

**`src/preprocessing/pipeline.py`**
- Implements `PreprocessingPipeline.run()` with stages: missing values (ffill→0), wavelet denoising, constant removal, metric clustering, high-correlation filtering, intelligent differencing
- Protected metrics (root causes) are preserved during clustering
- Saves correlation matrix to `data/processed/`

**`src/algorithms/factory.py`**
- Algorithm factory creating PC algorithm instances
- **Custom CI test**: Filters conditioning set variables by level hierarchy (prevents conditioning on higher-level nodes)
- Tracks all independence test results for debugging
- Outputs adjacency matrix, skeleton graph, separation sets

**`src/knowledge/knowledge_manager.py`**
- `KnowledgeManager` parses constraints.yaml to generate background knowledge
- **Hierarchy constraints**: Uses regex patterns to classify metrics into Resource (Level 1) → QoS (Level 2) → Business (Level 3); forbids high→low edges
- Supports explicit constraint patterns with regex

**`src/orientation/cascade_orientator.py`**
- `CascadeOrientator` orients undirected edges from PC output
- Priority order: Meek rules → Time lag (cross-correlation) → IGCI → V-structure preservation

**`src/rca/rca_engine.py`**
- Implements RADICE methodology for root cause localization
- Uses time-shift adjusted correlation for causal link strength
- Finds root causes by analyzing ancestor paths to symptom node

### Path Utilities

All data paths use variable substitution in config files: `{dataset}`, `{case}`, `{sample_name}`. Never hardcode paths.

## Important Implementation Details

### Level-Based Conditioning Set Filtering
In `pc_algorithm.py`, the custom CI test excludes variables from conditioning sets if their level > both X and Y levels. This prevents "conditioning on colliders" - a key assumption in this system's approach to causal discovery.

### Time Series Handling
- If `timestamp_column` is set in config, that column becomes the DataFrame index
- Time windows are applied via slicing on this index
- Preprocessed data (stationary) is used for time lag analysis; original data used for conditional entropy

### Metric Naming Conventions
The system expects metric names to match regex patterns in `constraints.yaml`. Common patterns:
- Resource: `cpu`, `memory`, `disk`, `gpu`, `io`, `network`, `inference`
- QoS: `latency`, `rt`, `duration`, `time`
- Business: `revenue`, `top`, metrics with numeric IDs like "0", "1", etc.

### Adding New Algorithms
1. Create new class in `src/algorithms/` inheriting pattern from `pc_algorithm.py`
2. Register in `AlgorithmFactory._algorithms` dict
3. Add corresponding config template in `config/templates/`

### Evaluation Format
Ground truth JSON expects:
```json
{
  "nodes": ["metric_a", "metric_b"],
  "edges": [["metric_a", "metric_b"]]  // a -> b
}
```

## Common Operations

**Run with different algorithm:**
```yaml
# config/config.yaml
algorithm:
  name: "pcmci"  # or "pc"
```

**Add protected metrics (prevent clustering removal):**
```yaml
# config/config.yaml
data:
  protected_metrics: ["cpu_usage", "memory_usage"]
```

**Add explicit causal constraints:**
```yaml
# config/constraints.yaml
explicit_required:
  - ['.*cpu.*', '.*latency.*']  # cpu must cause latency
explicit_forbidden:
  - ['.*revenue.*', '.*cpu.*']  # revenue cannot cause cpu
```