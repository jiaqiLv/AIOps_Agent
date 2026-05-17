# src Module Structure

This document describes the organization and purpose of each module in the causal discovery system.

## Module Overview

```
src/
├── io_utils/          # Data loading utilities (input/output)
├── preprocessing/      # Data preprocessing pipeline
├── knowledge/          # Background knowledge and constraints
├── algorithms/         # Causal discovery algorithms (PC)
├── orientation/        # Edge orientation (cascade)
├── rca/               # Root cause analysis (RADICE)
├── evaluation/        # Graph and RCA evaluation
├── output/            # Result saving
├── utils/             # Utility functions
├── conf/              # Configuration management
├── main.py            # Main pipeline entry point
└── batch_processor.py # Batch processing for RADICE datasets
```

**Note**: Module names `io_utils` and `conf` are used instead of `data` and `config` to avoid confusion with the top-level project directories.

## Module Details

### 1. io_utils/ - Data Loading Utilities Module (Input/Output)
**Purpose**: Load time series data from various formats

**Components**:
- `container.py` - `DataContainer`: Unified storage for metric data with metadata
- `loaders/base_loader.py` - `BaseLoader`: Abstract base for data loaders
- `loaders/csv_loader.py` - `CSVDataLoader`: Load from CSV files
- `loaders/radice_loader.py` - `RADICELoader`: Load RADICE format datasets

**RADICE Dataset Structure**:
```
data/raw/RADICE/{N5,N10,N15,N25}/artificialResults_*/
├── data.csv         # Time series data
├── edges.txt        # Ground truth causal edges
├── layer.txt        # Level constraints
├── root_cause.txt   # Root causes and symptoms
└── subgraph.txt     # Fault propagation subgraph
```

### 2. preprocessing/ - Preprocessing Module
**Purpose**: Prepare data for causal discovery

**Components**:
- `pipeline.py` - `PreprocessingPipeline`: Main preprocessing orchestration
- `registry.py` - `ProcessorRegistry`: Registry for preprocessing steps
- `processors/`:
  - `missing_values.py` - Handle missing values (ffill, zero)
  - `wavelet_denoise.py` - Wavelet-based denoising
  - `constant_filter.py` - Remove constant metrics
  - `correlation_filter.py` - Filter highly correlated metrics

### 3. knowledge/ - Background Knowledge Module
**Purpose**: Generate domain knowledge constraints

**Components**:
- `knowledge_manager.py` - `KnowledgeManager`: Load and manage constraints
- `classifier.py` - `MetricClassifier`: Classify metrics by type
- `constraint_builder.py` - `ConstraintBuilder`: Build forbidden/required edges
- `background_knowledge.py` - `BackgroundKnowledge`: Store knowledge for orientation

**Constraint Types**:
- Forbidden edges (high→low level)
- Required edges (known causal relationships)
- Level constraints (Resource→QoS→Business)

### 4. algorithms/ - Causal Discovery Algorithms
**Purpose**: Implement causal discovery algorithms

**Components**:
- `base_algorithm.py` - `BaseAlgorithm`: Abstract base for algorithms
- `pc_algorithm.py` - `PCAlgorithm`: PC algorithm with custom CI test
- `factory.py` - `AlgorithmFactory`: Create algorithm instances

**PC Algorithm Features**:
- Custom CI test with level-based conditioning set filtering
- Support for forbidden/required edges
- Background knowledge integration
- Separation set tracking

### 5. orientation/ - Edge Orientation Module
**Purpose**: Orient undirected edges from PC algorithm

**Components**:
- `cascade_orientator.py` - `CascadeOrientator`: Main cascade orientation with Meek rules, time lag, and IGCI
- `orientation_tracker.py` - `OrientationTracker`: Track orientation decisions

**Orientation Priority**:
1. Background knowledge (highest - applied in PC algorithm)
2. Meek rules (R0-R4 - from causal-learn)
3. Time lag + correlation threshold
4. IGCI (lowest)

### 6. rca/ - Root Cause Analysis Module
**Purpose**: Localize root causes using RADICE methodology

**Components**:
- `rca_engine.py` - `RCAEngine`: Main RCA engine
- `correlation_adjustment.py` - Adjusted correlation computation
- `graph_subtraction.py` - Extract causal subgraph

**RCA Workflow**:
1. Compute adjusted correlation scores
2. Filter candidates by score
3. Extract causal subgraph
4. Classify nodes (performance/root cause/intermediate)

### 7. evaluation/ - Evaluation Module
**Purpose**: Evaluate causal graphs and RCA results

**Components**:
- `evaluator.py` - `EvaluationEngine`: Main evaluation orchestrator
- `graph_evaluator.py` - `GraphEvaluator`: Graph metrics (F1, SHD)
- `rca_evaluator.py` - `RCAEvaluator`: RCA metrics (precision, recall)
- `metrics.py` - Metric computation functions

**Metrics**:
- Edge precision, recall, F1
- Structural Hamming Distance (SHD)
- Skeleton F1
- RCA precision, recall, F1

### 8. output/ - Result Saving Module
**Purpose**: Save analysis results

**Components**:
- `result_saver.py` - `ResultSaver`: Save all results

**Output Structure**:
```
data/output/{dataset}/{case}/{sample}/
├── final_graph.csv           # Oriented causal graph
└── intermediate/             # Algorithm intermediates

data/rca_result/{dataset}/{case}/{sample}/
├── evaluation_results.json   # Evaluation metrics
├── root_cause.txt           # Root causes (RADICE format)
├── root_causes.csv          # Root causes list
├── adjusted_correlation.csv # Adjusted correlation scores
└── subgraph.csv             # Fault propagation subgraph
```

### 9. utils/ - Utility Module
**Purpose**: Common utility functions

**Components**:
- `logger.py` - Logging setup

### 10. conf/ - Configuration Module
**Purpose**: Manage YAML configuration files

**Components**:
- `config_manager.py` - `ConfigManager`: Load and access config

**Configuration Files** (located in top-level `config/` directory):
- `config/config.yaml` - Main configuration
- `config/constraints.yaml` - Domain knowledge constraints
- `config/templates/` - Configuration templates

**Note**: The `conf` module name is used to distinguish from the top-level `config/` directory which contains the actual configuration files.

## Main Pipeline

The main pipeline (`main.py`) orchestrates:
1. Load data (data loaders)
2. Preprocess (preprocessing pipeline)
3. Generate constraints (knowledge manager)
4. Run causal discovery (PC algorithm)
5. Orient edges (cascade orientator)
6. Root cause analysis (RADICE engine)
7. Evaluate (evaluation engine)
8. Save results (result saver)

## Batch Processing

For RADICE datasets, `batch_processor.py` processes all samples:
- Lists all `artificialResults_*` folders
- Runs full pipeline on each sample
- Aggregates results
- Computes summary statistics
