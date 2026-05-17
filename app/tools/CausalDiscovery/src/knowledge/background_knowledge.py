"""
Background knowledge representation for causal discovery.

This module provides:
1. A data structure to represent background knowledge
2. A builder to create causal-learn BackgroundKnowledge objects
"""

from typing import Dict, List, Tuple, Set, Any, Optional, Union
from dataclasses import dataclass, field
import logging

from .constraint_builder import ConstraintBuilder

logger = logging.getLogger(__name__)

# Try to import causal-learn components
try:
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge as CausalBackgroundKnowledge
    from causallearn.graph.GraphNode import GraphNode
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CausalBackgroundKnowledge = None
    GraphNode = None
    CAUSAL_LEARN_AVAILABLE = False
    logger.warning("causal-learn library not available. BackgroundKnowledgeBuilder will be disabled.")


@dataclass
class BackgroundKnowledge:
    """
    Background knowledge for causal discovery.

    Encodes domain knowledge about:
    - Forbidden edges (cannot exist)
    - Required edges (must exist)
    - Level assignments (hierarchical position)
    - Type classifications
    """

    forbidden_edges: Set[Tuple[str, str]] = field(default_factory=set)
    required_edges: Set[Tuple[str, str]] = field(default_factory=set)
    level_map: Dict[str, int] = field(default_factory=dict)
    type_map: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_forbidden(self, source: str, target: str) -> None:
        """Add a forbidden edge."""
        self.forbidden_edges.add((source, target))

    def add_required(self, source: str, target: str) -> None:
        """Add a required edge."""
        self.required_edges.add((source, target))

    def set_level(self, node: str, level: int) -> None:
        """Set the level of a node."""
        self.level_map[node] = level

    def set_type(self, node: str, metric_type: str) -> None:
        """Set the type of a node."""
        self.type_map[node] = metric_type

    def get_level(self, node: str, default: int = None) -> int:
        """Get the level of a node."""
        return self.level_map.get(node, default)

    def get_type(self, node: str, default: str = None) -> str:
        """Get the type of a node."""
        return self.type_map.get(node, default)

    def is_forbidden(self, source: str, target: str) -> bool:
        """Check if an edge is forbidden."""
        return (source, target) in self.forbidden_edges

    def is_required(self, source: str, target: str) -> bool:
        """Check if an edge is required."""
        return (source, target) in self.required_edges

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the background knowledge."""
        return {
            'forbidden_edges_count': len(self.forbidden_edges),
            'required_edges_count': len(self.required_edges),
            'nodes_with_level': len(self.level_map),
            'nodes_with_type': len(self.type_map),
            'level_distribution': self._get_level_distribution(),
            'type_distribution': self._get_type_distribution()
        }

    def _get_level_distribution(self) -> Dict[int, int]:
        """Get distribution of nodes across levels."""
        distribution = {}
        for level in self.level_map.values():
            distribution[level] = distribution.get(level, 0) + 1
        return distribution

    def _get_type_distribution(self) -> Dict[str, int]:
        """Get distribution of nodes across types."""
        distribution = {}
        for mtype in self.type_map.values():
            distribution[mtype] = distribution.get(mtype, 0) + 1
        return distribution

    @classmethod
    def from_constraints(cls, forbidden: List[Tuple[str, str]],
                        required: List[Tuple[str, str]],
                        level_map: Dict[str, int],
                        type_map: Dict[str, str] = None) -> 'BackgroundKnowledge':
        """
        Create BackgroundKnowledge from constraint lists.

        Args:
            forbidden: List of forbidden (source, target) edges
            required: List of required (source, target) edges
            level_map: Dictionary mapping nodes to levels
            type_map: Optional dictionary mapping nodes to types

        Returns:
            BackgroundKnowledge instance
        """
        return cls(
            forbidden_edges=set(forbidden),
            required_edges=set(required),
            level_map=level_map or {},
            type_map=type_map or {}
        )


class BackgroundKnowledgeBuilder:
    """
    Builder for creating causal-learn BackgroundKnowledge objects.

    This builder:
    1. Creates GraphNode objects for metric names
    2. Maps metrics to tiers using add_node_to_tier
    3. Loads explicit causal edges using add_required_by_node and add_forbidden_by_node
    4. Returns the BackgroundKnowledge object
    """

    def __init__(self):
        """Initialize the builder."""
        self.node_objects: Dict[str, Any] = {}

    def build_background_knowledge(
        self,
        metric_names: List[str],
        level_map: Dict[str, int],
        explicit_forbidden: List[List[str]] = None,
        explicit_required: List[List[str]] = None
    ) -> Optional[Any]:
        """
        Build a causal-learn BackgroundKnowledge object.

        Args:
            metric_names: List of metric names
            level_map: Dictionary mapping metric names to levels
            explicit_forbidden: List of [source_pattern, target_pattern] for forbidden edges
            explicit_required: List of [source_pattern, target_pattern] for required edges

        Returns:
            causal-learn BackgroundKnowledge object, or None if causal-learn is not available
        """
        if not CAUSAL_LEARN_AVAILABLE:
            logger.error("causal-learn library is not available. Cannot create BackgroundKnowledge.")
            return None

        # Create the BackgroundKnowledge object
        bk = CausalBackgroundKnowledge()

        # Create GraphNode objects for each metric
        node_objects = {}
        for name in metric_names:
            node_objects[name] = GraphNode(name)

        # Add tier constraints using add_node_to_tier
        for name, level in level_map.items():
            if name in node_objects:
                node = node_objects[name]
                bk.add_node_to_tier(node, level)
                logger.debug(f"Added {name} to tier {level}")

        # Compile and add explicit forbidden edges
        if explicit_forbidden:
            compiled_forbidden = ConstraintBuilder.compile_constraint_patterns(explicit_forbidden)
            for src_pattern, dst_pattern in compiled_forbidden:
                matched_src = [n for n in metric_names if src_pattern.search(n.lower())]
                matched_dst = [n for n in metric_names if dst_pattern.search(n.lower())]

                for src in matched_src:
                    for dst in matched_dst:
                        if src in node_objects and dst in node_objects and src != dst:
                            bk.add_forbidden_by_node(node_objects[src], node_objects[dst])
                            logger.debug(f"Added forbidden edge: {src} -> {dst}")

        # Compile and add explicit required edges
        if explicit_required:
            compiled_required = ConstraintBuilder.compile_constraint_patterns(explicit_required)
            for src_pattern, dst_pattern in compiled_required:
                matched_src = [n for n in metric_names if src_pattern.search(n.lower())]
                matched_dst = [n for n in metric_names if dst_pattern.search(n.lower())]

                for src in matched_src:
                    for dst in matched_dst:
                        if src in node_objects and dst in node_objects and src != dst:
                            bk.add_required_by_node(node_objects[src], node_objects[dst])
                            logger.debug(f"Added required edge: {src} -> {dst}")

        logger.info(f"Created BackgroundKnowledge with {len(level_map)} tier assignments, "
                   f"{len(explicit_forbidden or [])} forbidden patterns, "
                   f"{len(explicit_required or [])} required patterns")
        return bk