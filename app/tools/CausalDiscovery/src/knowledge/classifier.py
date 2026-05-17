"""
Metric classifier for categorizing metrics by type using regex patterns.

This module provides functionality to classify metrics into types
(e.g., Resource, QoS, Business) based on regex pattern matching.
"""

import re
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of metric classification."""
    metric_name: str
    metric_type: str
    level: Optional[int] = None
    matched_pattern: Optional[str] = None


class MetricClassifier:
    """
    Classifies metrics into types based on regex patterns.

    Metric types are mapped to levels (integers) that represent
    hierarchical positions in the system architecture.
    """

    def __init__(self, metric_type_definitions: Dict[str, List[str]],
                 level_definitions: Dict[str, int]):
        """
        Initialize the metric classifier.

        Args:
            metric_type_definitions: Dictionary mapping type names to regex patterns
                e.g., {"Resource (资源层)": [".*cpu.*", ".*memory.*"]}
            level_definitions: Dictionary mapping type names to level IDs
                e.g., {"Resource (资源层)": 1}
        """
        self.metric_type_definitions = metric_type_definitions
        self.level_definitions = level_definitions

        # Pre-compile regex patterns for efficiency
        self.compiled_patterns: Dict[str, List[re.Pattern]] = {}
        for metric_type, patterns in metric_type_definitions.items():
            self.compiled_patterns[metric_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

        logger.info(f"Initialized classifier with {len(metric_type_definitions)} metric types")

    def classify(self, metric_name: str) -> ClassificationResult:
        """
        Classify a single metric.

        Args:
            metric_name: Name of the metric to classify

        Returns:
            ClassificationResult with type and level information
        """
        metric_name_lower = metric_name.lower()

        for metric_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(metric_name_lower):
                    level = self.level_definitions.get(metric_type)
                    return ClassificationResult(
                        metric_name=metric_name,
                        metric_type=metric_type,
                        level=level,
                        matched_pattern=pattern.pattern
                    )

        # No match found
        return ClassificationResult(
            metric_name=metric_name,
            metric_type='Unknown',
            level=None
        )

    def classify_batch(self, metric_names: List[str]) -> Dict[str, ClassificationResult]:
        """
        Classify multiple metrics.

        Args:
            metric_names: List of metric names to classify

        Returns:
            Dictionary mapping metric names to ClassificationResults
        """
        results = {}
        for name in metric_names:
            results[name] = self.classify(name)
        return results

    def get_metrics_by_type(self, classifications: Dict[str, ClassificationResult],
                            metric_type: str) -> List[str]:
        """
        Get all metrics of a specific type.

        Args:
            classifications: Dictionary of classification results
            metric_type: Type to filter by

        Returns:
            List of metric names of the specified type
        """
        return [
            name for name, result in classifications.items()
            if result.metric_type == metric_type
        ]

    def get_metrics_by_level(self, classifications: Dict[str, ClassificationResult],
                             level: int) -> List[str]:
        """
        Get all metrics at a specific level.

        Args:
            classifications: Dictionary of classification results
            level: Level to filter by

        Returns:
            List of metric names at the specified level
        """
        return [
            name for name, result in classifications.items()
            if result.level == level
        ]

    def get_level_distribution(self, classifications: Dict[str, ClassificationResult]) -> Dict[str, int]:
        """
        Get distribution of metrics across levels.

        Args:
            classifications: Dictionary of classification results

        Returns:
            Dictionary mapping level names to counts
        """
        distribution = {}
        for result in classifications.values():
            level_key = f"Level{result.level}" if result.level is not None else "Unknown"
            distribution[level_key] = distribution.get(level_key, 0) + 1
        return distribution

    def get_type_distribution(self, classifications: Dict[str, ClassificationResult]) -> Dict[str, int]:
        """
        Get distribution of metrics across types.

        Args:
            classifications: Dictionary of classification results

        Returns:
            Dictionary mapping type names to counts
        """
        distribution = {}
        for result in classifications.values():
            distribution[result.metric_type] = distribution.get(result.metric_type, 0) + 1
        return distribution

    def create_level_map(self, metric_names: List[str]) -> Dict[str, int]:
        """
        Create a level map dictionary for metrics.

        Args:
            metric_names: List of metric names

        Returns:
            Dictionary mapping metric names to levels
        """
        classifications = self.classify_batch(metric_names)
        return {
            name: result.level for name, result in classifications.items()
            if result.level is not None
        }
