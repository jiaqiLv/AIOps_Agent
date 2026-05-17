"""
Data container for storing time series data with metadata.

This module provides a unified container for time series data that includes
the metric data, timestamp information, and associated metadata.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataContainer:
    """
    Container for time series data with metadata.

    Attributes:
        metric_data: DataFrame containing metric time series data
        timestamp_column: Name of the timestamp column (if any)
        timestamp_index: Whether timestamp is in the index
        metadata: Additional metadata dictionary
    """

    metric_data: pd.DataFrame
    timestamp_column: Optional[str] = None
    timestamp_index: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the data container after initialization."""
        if self.metric_data is None or self.metric_data.empty:
            raise ValueError("metric_data cannot be None or empty")

        # Detect timestamp in index if not specified
        if self.timestamp_index is None:
            self.timestamp_index = self._detect_timestamp_index()

    def _detect_timestamp_index(self) -> bool:
        """Detect if the DataFrame index is a timestamp."""
        if hasattr(self.metric_data.index, 'name'):
            index_name_lower = str(self.metric_data.index.name).lower() if self.metric_data.index.name else ''
            return any(keyword in index_name_lower for keyword in ['time', 'date', 'timestamp'])
        return False

    def get_metric_names(self) -> List[str]:
        """
        Get list of metric column names.

        Returns:
            List of metric names (excluding timestamp column if present)
        """
        columns = self.metric_data.columns.tolist()
        if self.timestamp_column and self.timestamp_column in columns:
            columns.remove(self.timestamp_column)
        return columns

    def get_metric_data(self) -> pd.DataFrame:
        """
        Get the metric data (excluding timestamp column if present).

        Returns:
            DataFrame containing only metric columns
        """
        columns = self.get_metric_names()
        return self.metric_data[columns]

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the data container.

        Returns:
            Dictionary containing summary information
        """
        return {
            'n_samples': len(self.metric_data),
            'n_metrics': len(self.get_metric_names()),
            'metric_names': self.get_metric_names(),
            'timestamp_column': self.timestamp_column,
            'timestamp_index': self.timestamp_index,
            'has_missing_values': self.metric_data.isnull().any().any(),
            'missing_value_count': int(self.metric_data.isnull().sum().sum()),
            'metadata': self.metadata
        }

    def __repr__(self) -> str:
        """String representation."""
        return (f"DataContainer(n_samples={len(self.metric_data)}, "
                f"n_metrics={len(self.get_metric_names())}, "
                f"timestamp={'Yes' if self.timestamp_column or self.timestamp_index else 'No'})")