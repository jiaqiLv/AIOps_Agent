"""
Base processor interface for data preprocessing.

This module defines the abstract interface that all preprocessing processors must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BaseProcessor(ABC):
    """
    Abstract base class for data preprocessing processors.

    All processors must inherit from this class and implement the process() method.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the processor with configuration.

        Args:
            config: Configuration dictionary for the processor
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process the input data.

        Args:
            data: Input DataFrame

        Returns:
            Processed DataFrame
        """
        pass

    def validate_input(self, data: pd.DataFrame) -> None:
        """
        Validate input data.

        Args:
            data: Input DataFrame

        Raises:
            ValueError: If validation fails
        """
        if data is None or data.empty:
            raise ValueError("Input data cannot be None or empty")

    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the processor configuration.

        Returns:
            Dictionary with configuration summary
        """
        return {
            'enabled': self.enabled,
            'config': self.config
        }
