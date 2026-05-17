"""
Missing values processor for handling NaN values in time series data.

This processor fills missing values using forward fill followed by zero fill.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from .base_processor import BaseProcessor


class MissingValuesProcessor(BaseProcessor):
    """
    Processor for handling missing values in time series data.

    Uses a two-step approach:
    1. Forward fill (ffill) - propagate last valid value forward
    2. Zero fill - replace remaining NaN values with 0
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the missing values processor.

        Args:
            config: Configuration dictionary with optional keys:
                - enabled: Whether to enable this processor (default: true)
                - method: Method to use (default: "ffill_then_zero")
        """
        super().__init__(config)
        self.method = config.get('method', 'ffill_then_zero')

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process missing values in the data.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with missing values filled
        """
        if not self.enabled:
            return data

        self.validate_input(data)

        initial_nan_count = data.isnull().sum().sum()
        self.logger.info(f"Starting missing values handling. Initial NaN count: {initial_nan_count}")

        if initial_nan_count == 0:
            self.logger.info("No missing values found, skipping")
            return data

        df_processed = data.copy()

        if self.method == 'ffill_then_zero':
            # Step 1: Forward fill
            df_processed = df_processed.ffill()
            nan_after_ffill = df_processed.isnull().sum().sum()
            self.logger.info(f"After forward fill: {nan_after_ffill} NaN values remaining")

            # Step 2: Fill remaining NaN with 0
            df_processed = df_processed.fillna(0)
            final_nan_count = df_processed.isnull().sum().sum()
            self.logger.info(f"After zero fill: {final_nan_count} NaN values remaining")

        else:
            self.logger.warning(f"Unknown method '{self.method}', using default ffill_then_zero")
            df_processed = df_processed.ffill().fillna(0)

        self.logger.info(f"Missing values handling complete. Filled {initial_nan_count} values")

        return df_processed
