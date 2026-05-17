"""Knowledge management module."""
from .knowledge_manager import KnowledgeManager
from .background_knowledge import BackgroundKnowledgeBuilder

# Internal components (available but not primary API)
from .classifier import MetricClassifier
from .constraint_builder import ConstraintBuilder
from .background_knowledge import BackgroundKnowledge

__all__ = [
    # Primary API
    'KnowledgeManager',
    'BackgroundKnowledgeBuilder',
    # Internal components (exported for advanced use cases)
    'MetricClassifier',
    'ConstraintBuilder',
    'BackgroundKnowledge',
]
