"""
CSV data loader for time series data.

This module provides functionality to load time series data from CSV files
with support for time-based filtering and timestamp handling.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
import logging

from src.dataloader.loaders.base_loader import BaseLoader
from src.dataloader.container import DataContainer

logger = logging.getLogger(__name__)


class CSVDataLoader(BaseLoader):
    """
    Data loader for CSV files containing time series metrics.

    Supports:
    - Loading metric data from CSV files (first row is header)
    - Time-based filtering with optional time windows
    - Index-based filtering (for data without timestamp column)
    - Timestamp column handling (as column or index)
    - Variable substitution in file paths

    Time Window Filtering Modes:
        1. Timestamp mode (when timestamp_column is configured):
           - Uses start/end as datetime values for filtering
           - Example: time_window: {start: "2024-01-01 00:00:00", end: "2024-01-01 01:00:00"}

        2. Index mode (when no timestamp_column):
           - Uses start/end as integer row indices
           - Example: time_window: {start: 100, end: 500}
           - Supports negative indexing: -1 for last row, -10 for 10th from last
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the CSV data loader.

        Args:
            config: Configuration dictionary containing:
                - type: Must be "csv"
                - path: Path to the CSV file (already resolved with placeholders substituted)
                - timestamp_column: Name of the timestamp column (optional)
                - time_window: Optional dict with 'start' and 'end' bounds
                               - With timestamp_column: datetime strings
                               - Without timestamp_column: integer indices
        """
        super().__init__(config)
        # Path is already resolved by ConfigManager
        self.data_path = config.get('path', '')
        self.timestamp_column = config.get('timestamp_column', None)
        self.time_window = config.get('time_window', None)

        logger.info(f"Initialized CSVDataLoader with path: {self.data_path}")

    def load(self) -> DataContainer:
        """
        Load data from the configured CSV file.

        Returns:
            DataContainer with the loaded data

        Raises:
            FileNotFoundError: If the CSV file doesn't exist
        """
        # Path is already resolved by ConfigManager
        file_path = self.data_path

        if not Path(file_path).exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        logger.info(f"Loading data from: {file_path}")

        # Read CSV file (first row is header)
        df = pd.read_csv(file_path)

        logger.info(f"Loaded raw data: shape={df.shape}, columns={list(df.columns)}")

        # Handle timestamp column
        timestamp_index = False
        if self.timestamp_column and self.timestamp_column in df.columns:
            df[self.timestamp_column] = pd.to_datetime(df[self.timestamp_column])
            # Set timestamp as index for easier filtering
            df = df.set_index(self.timestamp_column)
            timestamp_index = True
            logger.info(f"Set '{self.timestamp_column}' as index")

        # Create data container
        container = DataContainer(
            metric_data=df,
            timestamp_column=self.timestamp_column,
            timestamp_index=timestamp_index,
            metadata={'source_path': file_path}
        )

        # Apply time window if configured
        if self.time_window:
            container = self.apply_time_window(container)
            logger.info(f"Applied time window: {container.metric_data.shape[0]} samples")

        return container

    def apply_time_window(self, container: DataContainer) -> DataContainer:
        """
        Apply time window filter to the data container.

        Supports two modes:
        1. Timestamp mode: When timestamp_column is configured, filters by datetime range
        2. Index mode: When no timestamp column, uses start/end as integer row indices

        Args:
            container: Input DataContainer

        Returns:
            Filtered DataContainer

        Raises:
            ValueError: If time_window configuration is invalid
        """
        if not self.time_window:
            return container

        start_time = self.time_window.get('start')
        end_time = self.time_window.get('end')

        if start_time is None and end_time is None:
            return container

        # Determine filter mode based on whether timestamp column exists
        has_timestamp = container.timestamp_index or (
            container.timestamp_column and container.timestamp_column in container.metric_data.columns
        )

        filtered_data = container.metric_data.copy()

        if has_timestamp:
            # Timestamp mode: Filter by datetime range
            # Convert to datetime if strings
            if start_time:
                start_time = pd.to_datetime(start_time)
            if end_time:
                end_time = pd.to_datetime(end_time)

            logger.info(f"Applying time window (timestamp mode): {start_time} to {end_time}")

            if container.timestamp_index:
                # Timestamp is in index
                if start_time:
                    filtered_data = filtered_data[filtered_data.index >= start_time]
                if end_time:
                    filtered_data = filtered_data[filtered_data.index <= end_time]
            else:
                # Timestamp is in column
                if start_time:
                    filtered_data = filtered_data[filtered_data[container.timestamp_column] >= start_time]
                if end_time:
                    filtered_data = filtered_data[filtered_data[container.timestamp_column] <= end_time]
        else:
            # Index mode: Use start/end as integer row indices
            # Parse as integers
            start_idx = int(start_time) if start_time is not None else None
            end_idx = int(end_time) if end_time is not None else None

            logger.info(f"Applying time window (index mode): rows {start_idx} to {end_idx}")

            # Use iloc for integer-based indexing
            total_rows = len(filtered_data)

            # Validate indices
            if start_idx is not None and start_idx < 0:
                start_idx = max(0, total_rows + start_idx)  # Handle negative indexing
            if end_idx is not None and end_idx < 0:
                end_idx = max(0, total_rows + end_idx)  # Handle negative indexing

            # Apply index-based slicing
            if start_idx is not None and end_idx is not None:
                filtered_data = filtered_data.iloc[start_idx:end_idx]
            elif start_idx is not None:
                filtered_data = filtered_data.iloc[start_idx:]
            elif end_idx is not None:
                filtered_data = filtered_data.iloc[:end_idx]

        return DataContainer(
            metric_data=filtered_data,
            timestamp_column=container.timestamp_column,
            timestamp_index=container.timestamp_index,
            metadata=container.metadata.copy()
        )
