"""
Orientation tracker for recording edge orientation decisions.

This module provides functionality to track and record all edge orientation
decisions made during the orientation process.
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OrientationMethod(Enum):
    """Methods used to orient edges."""
    BACKGROUND_KNOWLEDGE = "background_knowledge"
    MEEK_RULE = "meek_rule"
    TIME_LAG = "time_lag"
    IGCI = "igci"
    UNDETERMINED = "undetermined"


@dataclass
class OrientationRecord:
    """Record of a single edge orientation decision."""
    edge: tuple  # (source, target)
    original_direction: Optional[str]  # 'i->j', 'j->i', or 'undirected'
    final_direction: str  # 'i->j' or 'j->i'
    method: OrientationMethod
    method_detail: Optional[str] = None  # e.g., 'R0', 'R1', etc. for Meek rules
    confidence: float = 1.0
    iteration: int = 0


class OrientationTracker:
    """
    Tracks all orientation decisions made during the orientation process.

    Records:
    - Which edges were oriented
    - What method was used
    - The iteration number
    - Confidence scores
    """

    def __init__(self):
        """Initialize the orientation tracker."""
        self.records: List[OrientationRecord] = []
        self.iteration: int = 0
        self.oriented_edges: Set[tuple] = set()

    def start_iteration(self) -> None:
        """Start a new orientation iteration."""
        self.iteration += 1
        logger.debug(f"Starting orientation iteration {self.iteration}")

    def record_orientation(self, edge: tuple, original_direction: Optional[str],
                          final_direction: str, method: OrientationMethod,
                          method_detail: Optional[str] = None,
                          confidence: float = 1.0) -> None:
        """
        Record an orientation decision.

        Args:
            edge: (node1, node2) tuple
            original_direction: Direction before orientation
            final_direction: Direction after orientation ('i->j' or 'j->i')
            method: Method used for orientation
            method_detail: Additional detail (e.g., specific rule)
            confidence: Confidence score (0-1)
        """
        record = OrientationRecord(
            edge=edge,
            original_direction=original_direction,
            final_direction=final_direction,
            method=method,
            method_detail=method_detail,
            confidence=confidence,
            iteration=self.iteration
        )

        self.records.append(record)
        self.oriented_edges.add(edge)

        logger.debug(f"Recorded orientation: {edge[0]} -> {edge[1]} via {method.value}")

    def get_records_by_iteration(self, iteration: int) -> List[OrientationRecord]:
        """Get all orientation records from a specific iteration."""
        return [r for r in self.records if r.iteration == iteration]

    def get_records_by_method(self, method: OrientationMethod) -> List[OrientationRecord]:
        """Get all orientation records using a specific method."""
        return [r for r in self.records if r.method == method]

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of orientation decisions."""
        method_counts = {}
        for record in self.records:
            method = record.method.value
            method_counts[method] = method_counts.get(method, 0) + 1

        return {
            'total_orientations': len(self.records),
            'unique_edges_oriented': len(self.oriented_edges),
            'total_iterations': self.iteration,
            'method_counts': method_counts,
            'average_confidence': sum(r.confidence for r in self.records) / len(self.records) if self.records else 0
        }

    def get_orientations_as_list(self) -> List[Dict[str, Any]]:
        """Get orientation records as a list of dictionaries."""
        return [
            {
                'edge': f"{r.edge[0]} -> {r.edge[1]}",
                'original_direction': r.original_direction,
                'final_direction': r.final_direction,
                'method': r.method.value,
                'method_detail': r.method_detail,
                'confidence': r.confidence,
                'iteration': r.iteration
            }
            for r in self.records
        ]

    def clear(self) -> None:
        """Clear all records and reset iteration counter."""
        self.records.clear()
        self.oriented_edges.clear()
        self.iteration = 0
