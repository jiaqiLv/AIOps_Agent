"""
Base data loader interface.

This module defines the abstract interface that all data loaders must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from src.dataloader.container import DataContainer


class BaseLoader(ABC):
    """
    Abstract base class for data loaders.

    All data loaders must inherit from this class and implement the load() method.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data loader with configuration.

        Args:
            config: Configuration dictionary for the loader
        """
        self.config = config

    @abstractmethod
    def load(self) -> DataContainer:
        """
        Load data and return a DataContainer.

        Returns:
            DataContainer with loaded data

        Raises:
            FileNotFoundError: If the data file doesn't exist
        """
        pass

    def apply_time_window(self, container: DataContainer) -> DataContainer:
        """
        Apply time window filter to the data container.

        Default implementation returns the container unchanged.
        Override in subclasses to implement actual filtering.

        Args:
            container: Input DataContainer

        Returns:
            Filtered DataContainer (or unchanged)
        """
        return container
