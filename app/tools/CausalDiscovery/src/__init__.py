"""
Refactored microservice causal discovery system v2.0

This package provides a modular, extensible framework for causal discovery
and root cause analysis in microservice architectures.

Modules:
- dataloader: Data loading utilities from various formats (CSV, RADICE)
- preprocessing: Data preprocessing pipeline
- knowledge: Background knowledge and constraints
- algorithms: Causal discovery algorithms (PC)
- orientation: Edge orientation (Meek, time-lag, IGCI)
- rca: Root cause analysis (RCA)
- evaluation: Graph and RCA evaluation
- output: Result saving
- utils: Utility functions and configuration management
"""

__version__ = "2.0.0"

from .dataloader import DataContainer, CSVDataLoader, RADICELoader
from .preprocessing import PreprocessingPipeline
from .knowledge import KnowledgeManager, BackgroundKnowledge
from .algorithms import AlgorithmFactory
from .orientation import CascadeOrientator
from .rca import RCAEngine, RCAConfig
from .evaluation import EvaluationEngine
from .output import ResultSaver
from .utils import ConfigManager

__all__ = [
    'DataContainer',
    'CSVDataLoader',
    'RADICELoader',
    'PreprocessingPipeline',
    'KnowledgeManager',
    'BackgroundKnowledge',
    'AlgorithmFactory',
    'CascadeOrientator',
    'RCAEngine',
    'RCAConfig',
    'EvaluationEngine',
    'ResultSaver',
    'ConfigManager'
]
