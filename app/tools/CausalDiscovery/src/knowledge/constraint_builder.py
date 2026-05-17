"""
Constraint builder for generating causal constraints from domain knowledge.

This module generates forbidden and required edge constraints based on:
1. Hierarchy rules (higher levels cannot point to lower levels)
2. Explicit pattern-based constraints
"""

import re
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import logging

from .classifier import MetricClassifier

logger = logging.getLogger(__name__)


@dataclass
class ConstraintResult:
    """Result of constraint generation."""
    forbidden_edges: List[Tuple[str, str]]
    required_edges: List[Tuple[str, str]]
    level_map: Dict[str, int]
    type_distribution: Dict[str, int]
    level_distribution: Dict[str, int]


class ConstraintBuilder:
    """
    Builds causal discovery constraints from domain knowledge.

    Generates two types of constraints:
    1. Forbidden edges: Edges that cannot exist in the causal graph
    2. Required edges: Edges that must exist in the causal graph
    """

    def __init__(self, metric_type_definitions: Dict[str, List[str]],
                 level_definitions: Dict[str, int],
                 explicit_forbidden: List[List[str]] = None,
                 explicit_required: List[List[str]] = None):
        """
        Initialize the constraint builder.

        Args:
            metric_type_definitions: Dictionary mapping type names to regex patterns
            level_definitions: Dictionary mapping type names to level IDs
            explicit_forbidden: List of [source_pattern, target_pattern] for forbidden edges
            explicit_required: List of [source_pattern, target_pattern] for required edges
        """
        self.metric_type_definitions = metric_type_definitions
        self.level_definitions = level_definitions
        self.explicit_forbidden = explicit_forbidden or []
        self.explicit_required = explicit_required or []

        # Initialize classifier
        self.classifier = MetricClassifier(
            metric_type_definitions,
            level_definitions
        )

        # Pre-compile explicit constraint patterns
        self.compiled_explicit_forbidden = ConstraintBuilder.compile_constraint_patterns(self.explicit_forbidden)
        self.compiled_explicit_required = ConstraintBuilder.compile_constraint_patterns(self.explicit_required)

    def build(self, metric_names: List[str]) -> ConstraintResult:
        """
        Build constraints for the given metrics.

        Args:
            metric_names: List of metric names to generate constraints for

        Returns:
            ConstraintResult with all constraints and metadata
        """
        logger.info(f"Building constraints for {len(metric_names)} metrics")

        # Classify all metrics
        classifications = self.classifier.classify_batch(metric_names)
        level_map = self.classifier.create_level_map(metric_names)

        # Generate hierarchy constraints
        forbidden_edges = self._generate_hierarchy_constraints(metric_names, level_map)

        # Add explicit constraints
        forbidden_edges.update(self._parse_explicit_constraints(
            self.compiled_explicit_forbidden, metric_names
        ))
        required_edges = self._parse_explicit_constraints(
            self.compiled_explicit_required, metric_names
        )

        # Get distributions
        type_dist = self.classifier.get_type_distribution(classifications)
        level_dist = self.classifier.get_level_distribution(classifications)

        logger.info(f"Generated {len(forbidden_edges)} forbidden, {len(required_edges)} required edges")
        logger.info(f"Type distribution: {type_dist}")
        logger.info(f"Level distribution: {level_dist}")

        return ConstraintResult(
            forbidden_edges=list(forbidden_edges),
            required_edges=list(required_edges),
            level_map=level_map,
            type_distribution=type_dist,
            level_distribution=level_dist
        )

    def _generate_hierarchy_constraints(self, metric_names: List[str],
                                       level_map: Dict[str, int]) -> Set[Tuple[str, str]]:
        """
        Generate hierarchy-based forbidden edges.

        Rule: Higher level (larger number) cannot point to lower level (smaller number).
        This prevents "effect -> cause" relationships in the hierarchy.

        Args:
            metric_names: List of metric names
            level_map: Dictionary mapping metric names to levels

        Returns:
            Set of forbidden (source, target) tuples
        """
        forbidden_edges = set()

        for src in metric_names:
            for dst in metric_names:
                if src == dst:
                    continue

                src_level = level_map.get(src)
                dst_level = level_map.get(dst)

                # If both have defined levels and src is higher than dst
                if src_level is not None and dst_level is not None:
                    if src_level > dst_level:
                        forbidden_edges.add((src, dst))
                        logger.debug(f"Hierarchy constraint: {src}(L{src_level}) -> {dst}(L{dst_level})")

        return forbidden_edges

    @staticmethod
    def compile_constraint_patterns(constraints: List[List[str]]) -> List[Tuple[re.Pattern, re.Pattern]]:
        """
        Compile regex patterns for explicit constraints.

        Args:
            constraints: List of [source_pattern, target_pattern]

        Returns:
            List of (compiled_source, compiled_target) tuples
        """
        compiled = []
        for constraint in constraints:
            if len(constraint) == 2:
                src_pattern, dst_pattern = constraint
                try:
                    compiled_src = re.compile(src_pattern, re.IGNORECASE)
                    compiled_dst = re.compile(dst_pattern, re.IGNORECASE)
                    compiled.append((compiled_src, compiled_dst))
                except re.error as e:
                    logger.warning(f"Invalid regex pattern: {src_pattern} -> {dst_pattern}: {e}")
        return compiled

    def _parse_explicit_constraints(self, compiled_constraints: List[Tuple[re.Pattern, re.Pattern]],
                                    metric_names: List[str]) -> Set[Tuple[str, str]]:
        """
        Parse explicit constraints against available metrics.

        Args:
            compiled_constraints: List of (source_pattern, target_pattern) compiled regex
            metric_names: Available metric names

        Returns:
            Set of constraint (source, target) tuples
        """
        constraint_edges = set()

        for src_pattern, dst_pattern in compiled_constraints:
            # Find matching metrics
            matched_src = [m for m in metric_names if src_pattern.search(m.lower())]
            matched_dst = [m for m in metric_names if dst_pattern.search(m.lower())]

            if not matched_src:
                logger.debug(f"Source pattern matched no metrics: {src_pattern.pattern}")
                continue
            if not matched_dst:
                logger.debug(f"Target pattern matched no metrics: {dst_pattern.pattern}")
                continue

            # Generate all constraint edges
            for src in matched_src:
                for dst in matched_dst:
                    if src != dst:  # Avoid self-loops
                        constraint_edges.add((src, dst))

        return constraint_edges
