"""
Preprocessing pipeline for orchestrating multiple data preprocessing steps.

This module provides a pipeline that orchestrates multiple processors in sequence
to clean and prepare data for causal discovery.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from pathlib import Path
import logging

from .registry import ProcessorRegistry

if TYPE_CHECKING:
    from .processors.base_processor import BaseProcessor

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Pipeline for data preprocessing.

    Orchestrates multiple processors in sequence to clean and prepare data.
    Processors are applied in the following order:
    1. Missing values handling - Fill NaN values using ffill then zero
    2. Constant filter - Remove columns with zero or near-zero variance
    3. Wavelet denoising - Apply wavelet transform-based denoising
    4. Correlation filter - Remove highly correlated redundant metrics
    """

    # Default processor order
    DEFAULT_PROCESSOR_ORDER = [
        'missing_values',
        'constant_filter',
        'wavelet_denoise',
        'correlation_filter',
    ]

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the preprocessing pipeline.

        Args:
            config: Configuration dictionary with 'processors' section
                    containing processor-specific configurations
        """
        self.config = config
        self.processors_config = config.get('processors', {})

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Initialize processors
        self.processors: List[BaseProcessor] = []
        self._initialize_processors()

        # Output configuration
        self.output_path = config.get('output_path', None)
        self.save_correlation = config.get('save_correlation_matrix', True)

        self.logger.info(f"Initialized preprocessing pipeline with {len(self.processors)} processors")
        self.logger.info(f"Output path from config: {self.output_path}")

    def _initialize_processors(self) -> None:
        """Initialize processor instances based on configuration."""
        for processor_name in self.DEFAULT_PROCESSOR_ORDER:
            # Get processor config (empty dict if not specified)
            processor_config = self.processors_config.get(processor_name, {})

            # Create processor instance
            try:
                processor = ProcessorRegistry.create(processor_name, processor_config)
                self.processors.append(processor)
                logger.debug(f"Added processor: {processor_name} (enabled={processor.enabled})")
            except Exception as e:
                logger.warning(f"Failed to create processor '{processor_name}': {e}")

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run the preprocessing pipeline on input data.

        Args:
            data: Input DataFrame

        Returns:
            Preprocessed DataFrame
        """
        self.logger.info(f"Starting preprocessing pipeline: input shape={data.shape}")
        self.logger.info(f"Output path configured: {self.output_path}")

        processed_data = data.copy()

        # Apply each processor in sequence
        for processor in self.processors:
            if not processor.enabled:
                continue

            processor_name = processor.__class__.__name__
            input_shape = processed_data.shape

            try:
                processed_data = processor.process(processed_data)
                self.logger.info(
                    f"{processor_name}: {input_shape} -> {processed_data.shape}"
                )
            except Exception as e:
                self.logger.error(f"{processor_name} failed: {e}")
                raise

        self.logger.info(f"Preprocessing pipeline complete: output shape={processed_data.shape}")

        # Save processed data and correlation matrix if configured
        if self.output_path:
            self.logger.info(f"Calling _save_results with output_path: {self.output_path}")
            self._save_results(processed_data)
        else:
            self.logger.warning("output_path is None, skipping save of processed data")

        return processed_data

    def _save_results(self, data: pd.DataFrame) -> None:
        """
        Save processed data and correlation matrix.

        Args:
            data: Preprocessed DataFrame
        """
        try:
            # output_path is already resolved by ConfigManager
            output_path = Path(self.output_path)
            self.logger.info(f"Attempting to save to: {output_path}")
            self.logger.info(f"Output path exists?: {output_path.parent.exists()}")

            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {output_path.parent}")

            # Save processed data
            data.to_csv(output_path, index=True)
            self.logger.info(f"Successfully saved processed data to: {output_path}")

            # Save correlation matrix
            if self.save_correlation:
                corr_output_path = output_path.parent / "pearson_correlation.csv"
                corr_matrix = data.corr(method='pearson')
                corr_matrix.to_csv(corr_output_path)
                self.logger.info(f"Successfully saved correlation matrix to: {corr_output_path}")

        except Exception as e:
            self.logger.error(f"Failed to save results: {e}", exc_info=True)
