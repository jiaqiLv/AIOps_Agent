"""
Algorithm factory for creating causal discovery algorithm instances.

This module provides a factory pattern for creating algorithm instances
based on configuration names.
"""

from typing import Dict, Type, Any
from .base_algorithm import BaseAlgorithm
from .pc_algorithm import PCAlgorithm
import logging

logger = logging.getLogger(__name__)


class AlgorithmFactory:
    """
    Factory for creating causal discovery algorithm instances.

    Supports dynamic registration of new algorithms.
    """

    # Default algorithm registry
    _algorithms: Dict[str, Type[BaseAlgorithm]] = {
        'pc': PCAlgorithm,
    }

    @classmethod
    def register(cls, name: str, algorithm_class: Type[BaseAlgorithm]) -> None:
        """
        Register a new algorithm class.

        Args:
            name: Name to register the algorithm under
            algorithm_class: Algorithm class to register
        """
        if not issubclass(algorithm_class, BaseAlgorithm):
            raise ValueError(f"{algorithm_class} must inherit from BaseAlgorithm")

        cls._algorithms[name] = algorithm_class
        logger.info(f"Registered algorithm: {name}")

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Unregister an algorithm class.

        Args:
            name: Name of the algorithm to unregister
        """
        if name in cls._algorithms:
            del cls._algorithms[name]
            logger.info(f"Unregistered algorithm: {name}")

    @classmethod
    def create(cls, name: str, **params) -> BaseAlgorithm:
        """
        Create an algorithm instance.

        Args:
            name: Name of the algorithm to create
            **params: Algorithm parameters

        Returns:
            Algorithm instance

        Raises:
            ValueError: If algorithm name is not registered
        """
        if name not in cls._algorithms:
            available = list(cls._algorithms.keys())
            raise ValueError(f"Unknown algorithm '{name}'. Available: {available}")

        algorithm_class = cls._algorithms[name]
        return algorithm_class(**params)

    @classmethod
    def list_algorithms(cls) -> list:
        """
        Get list of registered algorithm names.

        Returns:
            List of algorithm names
        """
        return list(cls._algorithms.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if an algorithm is registered.

        Args:
            name: Name of the algorithm

        Returns:
            True if registered, False otherwise
        """
        return name in cls._algorithms
