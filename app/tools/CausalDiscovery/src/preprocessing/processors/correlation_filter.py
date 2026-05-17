"""
Correlation filter processor for removing highly correlated metrics.

This processor identifies pairs of metrics with very high correlation and
removes the one with lower variance, keeping the more informative metric.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Set, Tuple
from .base_processor import BaseProcessor


class CorrelationFilterProcessor(BaseProcessor):
    """
    Processor for filtering out highly correlated metrics.

    For each pair of metrics with correlation above the threshold,
    removes the metric with lower variance.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the correlation filter processor.

        Args:
            config: Configuration dictionary with optional keys:
                - enabled: Whether to enable this processor (default: true)
                - threshold: Correlation threshold (default: 0.999)
        """
        super().__init__(config)
        self.threshold = config.get('threshold', 0.999)

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out highly correlated metrics.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with highly correlated metrics removed
        """
        if not self.enabled:
            return data

        self.validate_input(data)

        # Compute correlation matrix
        corr_matrix = data.corr().abs()

        # Find high correlation pairs
        high_corr_pairs = []
        columns_to_remove: Set[str] = set()

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]

                if corr_value > self.threshold:
                    high_corr_pairs.append((col1, col2, corr_value))

                    # Keep metric with higher variance
                    var1 = data[col1].var()
                    var2 = data[col2].var()

                    if var1 >= var2:
                        columns_to_remove.add(col2)
                    else:
                        columns_to_remove.add(col1)

        if high_corr_pairs:
            self.logger.info(f"Found {len(high_corr_pairs)} highly correlated pairs (threshold={self.threshold})")
            self.logger.info(f"Removing columns: {list(columns_to_remove)}")

            df_filtered = data.drop(columns=columns_to_remove)
            self.logger.info(f"Removed {len(columns_to_remove)} columns, {len(df_filtered.columns)} remaining")
        else:
            df_filtered = data.copy()
            self.logger.info(f"No highly correlated pairs found (threshold={self.threshold})")

        return df_filtered
