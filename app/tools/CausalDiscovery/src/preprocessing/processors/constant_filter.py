"""
Constant filter processor for removing constant columns from time series data.

This processor identifies and removes columns with zero variance (constant values).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from .base_processor import BaseProcessor


class ConstantFilterProcessor(BaseProcessor):
    """
    Processor for filtering out constant columns.

    A column is considered constant if its variance is zero (or very close to zero).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the constant filter processor.

        Args:
            config: Configuration dictionary with optional keys:
                - enabled: Whether to enable this processor (default: true)
                - variance_threshold: Minimum variance threshold (default: 0)
        """
        super().__init__(config)
        self.variance_threshold = config.get('variance_threshold', 0)

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out constant columns from the data.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with constant columns removed
        """
        if not self.enabled:
            return data

        self.validate_input(data)

        # Calculate variance for each column
        variances = data.var()

        # Identify constant columns
        constant_columns = variances[variances <= self.variance_threshold].index.tolist()

        if constant_columns:
            self.logger.info(f"Found {len(constant_columns)} constant columns: {constant_columns}")
            df_filtered = data.drop(columns=constant_columns)
            self.logger.info(f"Removed {len(constant_columns)} constant columns, {len(df_filtered.columns)} remaining")
        else:
            df_filtered = data.copy()
            self.logger.info("No constant columns found")

        return df_filtered
