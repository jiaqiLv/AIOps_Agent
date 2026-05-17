"""
Processor registry for managing available preprocessing processors.

This module provides a registry pattern for managing and creating processor instances.
"""

from typing import Dict, Type, Any
from .processors.base_processor import BaseProcessor
from .processors.missing_values import MissingValuesProcessor
from .processors.wavelet_denoise import WaveletDenoiseProcessor
from .processors.constant_filter import ConstantFilterProcessor
from .processors.correlation_filter import CorrelationFilterProcessor
import logging

logger = logging.getLogger(__name__)


class ProcessorRegistry:
    """
    Registry for managing available preprocessing processors.

    This class implements a registry pattern that allows dynamic registration
    and creation of processor instances.
    """

    # Default processor registry
    _processors: Dict[str, Type[BaseProcessor]] = {
        'missing_values': MissingValuesProcessor,
        'wavelet_denoise': WaveletDenoiseProcessor,
        'constant_filter': ConstantFilterProcessor,
        'correlation_filter': CorrelationFilterProcessor,
    }

    @classmethod
    def register(cls, name: str, processor_class: Type[BaseProcessor]) -> None:
        """
        Register a new processor class.

        Args:
            name: Name to register the processor under
            processor_class: Processor class to register
        """
        if not issubclass(processor_class, BaseProcessor):
            raise ValueError(f"{processor_class} must inherit from BaseProcessor")

        cls._processors[name] = processor_class
        logger.info(f"Registered processor: {name}")

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Unregister a processor class.

        Args:
            name: Name of the processor to unregister
        """
        if name in cls._processors:
            del cls._processors[name]
            logger.info(f"Unregistered processor: {name}")

    @classmethod
    def create(cls, name: str, config: Dict[str, Any]) -> BaseProcessor:
        """
        Create a processor instance.

        Args:
            name: Name of the processor to create
            config: Configuration dictionary for the processor

        Returns:
            Processor instance

        Raises:
            ValueError: If processor name is not registered
        """
        if name not in cls._processors:
            available = list(cls._processors.keys())
            raise ValueError(f"Unknown processor '{name}'. Available: {available}")

        processor_class = cls._processors[name]
        return processor_class(config)

    @classmethod
    def list_processors(cls) -> list:
        """
        Get list of registered processor names.

        Returns:
            List of processor names
        """
        return list(cls._processors.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a processor is registered.

        Args:
            name: Name of the processor

        Returns:
            True if registered, False otherwise
        """
        return name in cls._processors
