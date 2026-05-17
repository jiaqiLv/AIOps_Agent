"""
Wavelet denoising processor for removing noise from time series data.

This processor applies wavelet transform-based denoising to reduce noise
while preserving important signal features.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base_processor import BaseProcessor

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False


class WaveletDenoiseProcessor(BaseProcessor):
    """
    Processor for wavelet-based denoising of time series data.

    Uses discrete wavelet transform (DWT) to decompose the signal,
    applies thresholding to detail coefficients, and reconstructs the signal.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the wavelet denoise processor.

        Args:
            config: Configuration dictionary with optional keys:
                - enabled: Whether to enable this processor (default: true)
                - wavelet: Wavelet type to use (default: "db4")
                - level: Decomposition level (default: 1)
        """
        super().__init__(config)
        self.wavelet = config.get('wavelet', 'db4')
        self.level = config.get('level', 1)

        if not PYWT_AVAILABLE:
            self.logger.warning("PyWavelets not installed, wavelet denoising will be disabled")

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply wavelet denoising to the data.

        Args:
            data: Input DataFrame

        Returns:
            Denoised DataFrame
        """
        if not self.enabled or not PYWT_AVAILABLE:
            return data

        self.validate_input(data)

        df_denoised = data.copy()

        for col in data.columns:
            ts = data[col].values

            # Only process if data length is sufficient
            if len(ts) <= 2 ** (self.level + 1):
                self.logger.debug(f"Column '{col}': insufficient data length ({len(ts)}), skipping")
                df_denoised[col] = ts
                continue

            # Apply wavelet denoising
            denoised = self._denoise_series(ts)
            df_denoised[col] = denoised[:len(ts)]  # Ensure same length

        self.logger.info(f"Wavelet denoising complete: wavelet={self.wavelet}, level={self.level}")

        return df_denoised

    def _denoise_series(self, ts: np.ndarray) -> np.ndarray:
        """
        Apply wavelet denoising to a single time series.

        Args:
            ts: Input time series

        Returns:
            Denoised time series
        """
        try:
            # Decompose signal
            coeffs = pywt.wavedec(ts, self.wavelet, mode='per', level=self.level)

            # Compute threshold using median absolute deviation
            if len(coeffs) > 1:
                sigma = (1 / 0.6745) * np.median(np.abs(coeffs[-1] - np.median(coeffs[-1])))
                threshold = sigma * np.sqrt(2 * np.log(len(ts)))

                if threshold > 0 and not np.isnan(threshold):
                    # Apply soft thresholding to detail coefficients
                    new_coeffs = [coeffs[0]] + [
                        pywt.threshold(c, value=threshold, mode='soft')
                        for c in coeffs[1:]
                    ]
                else:
                    # Threshold is zero, keep original coefficients
                    new_coeffs = coeffs

                # Reconstruct signal
                reconstructed = pywt.waverec(new_coeffs, self.wavelet, mode='per')
            else:
                reconstructed = ts

            return reconstructed

        except Exception as e:
            self.logger.warning(f"Wavelet denoising failed: {e}, returning original signal")
            return ts
