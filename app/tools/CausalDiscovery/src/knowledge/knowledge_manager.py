"""
Knowledge manager for loading and managing domain knowledge.

This module provides the main interface for loading knowledge from YAML files
and generating background knowledge for causal discovery.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

from .classifier import MetricClassifier
from .constraint_builder import ConstraintBuilder
from .background_knowledge import BackgroundKnowledgeBuilder

logger = logging.getLogger(__name__)


class KnowledgeManager:
    """
    Manager for loading and processing domain knowledge.

    This class handles:
    1. Loading knowledge from YAML configuration files
    2. Classifying metrics by type
    3. Creating background knowledge objects for causal-learn
    """

    def __init__(self, constraints_path: Optional[str] = None,
                 config_dict: Optional[Dict] = None):
        """
        Initialize the knowledge manager.

        Args:
            constraints_path: Path to the constraints YAML file
            config_dict: Optional dictionary containing constraints configuration
        """
        self.config: Dict = {}
        self.constraints_path: Optional[str] = None

        if config_dict is not None:
            self.config = config_dict
        elif constraints_path is not None:
            self.load(constraints_path)

        # Initialize components
        self._init_components()

        # Initialize the background knowledge builder
        self.background_knowledge_builder = BackgroundKnowledgeBuilder()

    def load(self, constraints_path: str) -> 'KnowledgeManager':
        """
        Load constraints from a YAML file.

        Args:
            constraints_path: Path to the constraints YAML file (already resolved)

        Returns:
            Self for method chaining
        """
        # Path is already resolved by ConfigManager
        self.constraints_path = constraints_path

        if not Path(constraints_path).exists():
            raise FileNotFoundError(f"Constraints file not found: {constraints_path}")

        with open(constraints_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f) or {}

        logger.info(f"Loaded constraints from: {constraints_path}")
        return self

    def _init_components(self) -> None:
        """Initialize classifier and constraint builder."""
        metric_types = self.config.get('metric_type_definitions', {})
        level_definitions = self.config.get('level_definitions', {})
        explicit_forbidden = self.config.get('explicit_forbidden', [])
        explicit_required = self.config.get('explicit_required', [])

        self.classifier = MetricClassifier(metric_types, level_definitions)
        self.constraint_builder = ConstraintBuilder(
            metric_types,
            level_definitions,
            explicit_forbidden,
            explicit_required
        )

    def get_level_map(self, metric_names: List[str]) -> Dict[str, int]:
        """
        Get level assignments for metrics.

        Args:
            metric_names: List of metric names

        Returns:
            Dictionary mapping metric names to levels
        """
        return self.classifier.create_level_map(metric_names)

    def create_causal_learn_background_knowledge(
        self,
        metric_names: List[str],
        level_map: Dict[str, int] = None,
        explicit_forbidden: List[List[str]] = None,
        explicit_required: List[List[str]] = None
    ) -> Optional[Any]:
        """
        Create a causal-learn BackgroundKnowledge object.

        This method:
        1. Creates GraphNode objects for each metric name
        2. Maps metrics to tiers using add_node_to_tier
        3. Loads explicit causal edges using add_required_by_node and add_forbidden_by_node
        4. Returns the BackgroundKnowledge object

        Args:
            metric_names: List of metric names
            level_map: Dictionary mapping metric names to levels (tier numbers)
            explicit_forbidden: List of [source_pattern, target_pattern] for forbidden edges
            explicit_required: List of [source_pattern, target_pattern] for required edges

        Returns:
            causal-learn BackgroundKnowledge object, or None if causal-learn is not available
        """
        # Use provided level_map or generate from classifier
        if level_map is None:
            level_map = self.get_level_map(metric_names)

        # Use config explicit constraints if not provided
        if explicit_forbidden is None:
            explicit_forbidden = self.config.get('explicit_forbidden', [])
        if explicit_required is None:
            explicit_required = self.config.get('explicit_required', [])

        # Build the BackgroundKnowledge object
        return self.background_knowledge_builder.build_background_knowledge(
            metric_names=metric_names,
            level_map=level_map,
            explicit_forbidden=explicit_forbidden,
            explicit_required=explicit_required
        )