#!/usr/bin/env python3
"""
Cross-Correlation Function Plot from CSV Data

This script reads two columns from CSV data and creates cross-correlation plots,
similar to scripts/plot_ccf.py but using real data from CSV files.
All titles and legends are in English.

Usage:
    python plot_ccf_from_csv.py <csv_file> <column1> <column2> [output_file]

Example:
    python plot_ccf_from_csv.py ../data/processed/zh_dataset/1116/data_processed.csv 15_host_CPU 15_mem_usage ccf_plot.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


def load_data_from_csv(csv_file, col1, col2):
    """
    Load two columns from CSV file
    
    Args:
        csv_file: Path to CSV file
        col1: First column name
        col2: Second column name
    
    Returns:
        tuple: (data1, data2, column_names)
    """
    try:
        df = pd.read_csv(csv_file)
        print(f"Successfully loaded CSV file: {csv_file}")
        print(f"Data shape: {df.shape}")
        
        if col1 not in df.columns:
            print(f"Error: Column '{col1}' not found")
            print(f"Available columns: {df.columns.tolist()}")
            sys.exit(1)
        
        if col2 not in df.columns:
            print(f"Error: Column '{col2}' not found")
            print(f"Available columns: {df.columns.tolist()}")
            sys.exit(1)
        
        # Extract data and remove missing values
        data1 = df[col1].dropna().values
        data2 = df[col2].dropna().values
        
        # Align data by taking the minimum length
        min_len = min(len(data1), len(data2))
        data1 = data1[:min_len]
        data2 = data2[:min_len]
        
        print(f"Using {len(data1)} data points for analysis")
        print(f"{col1} range: {np.min(data1):.3f} to {np.max(data1):.3f}")
        print(f"{col2} range: {np.min(data2):.3f} to {np.max(data2):.3f}")
        
        return data1, data2, (col1, col2)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def normalize_zscore(data):
    """
    Normalize data using z-score (zero mean, unit variance).

    Args:
        data: Input data array

    Returns:
        Normalized data array
    """
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return data - mean  # Avoid division by zero
    return (data - mean) / std


def compute_cross_correlation(data1, data2):
    """
    Compute cross-correlation function between two signals.
    Using the same accurate normalization as src/orientation/cascade_orientator.py.

    Args:
        data1: First signal array
        data2: Second signal array

    Returns:
        tuple: (ccf_values, lags, positive_peak_lag, positive_peak_value, negative_peak_lag, negative_peak_value, data1_norm, data2_norm)
    """
    from scipy.signal import correlate

    # Remove NaN values
    valid_mask = ~(np.isnan(data1) | np.isnan(data2))
    x_clean = data1[valid_mask]
    y_clean = data2[valid_mask]

    # Normalize to zero mean, unit variance (z-score)
    x_norm = normalize_zscore(x_clean)
    y_norm = normalize_zscore(y_clean)

    n = len(x_norm)

    # Compute cross-correlation
    cross_corr = correlate(y_norm, x_norm, mode='full')
    lags = np.arange(-n + 1, n)

    # Accurate normalization: divide by effective overlap at each lag
    # For lag k, the number of overlapping points is n - abs(k)
    overlaps = n - np.abs(lags)
    overlaps = np.maximum(overlaps, 1)  # Avoid division by zero
    ccf_values = cross_corr / overlaps

    # Find positive correlation peak
    positive_mask = ccf_values >= 0
    if np.any(positive_mask):
        positive_peak_idx = np.argmax(ccf_values * positive_mask)
        positive_peak_lag = lags[positive_peak_idx]
        positive_peak_value = ccf_values[positive_peak_idx]
    else:
        positive_peak_lag = 0
        positive_peak_value = 0

    # Find negative correlation peak (minimum value)
    negative_mask = ccf_values < 0
    if np.any(negative_mask):
        negative_peak_idx = np.argmin(ccf_values)
        negative_peak_lag = lags[negative_peak_idx]
        negative_peak_value = ccf_values[negative_peak_idx]
    else:
        negative_peak_lag = 0
        negative_peak_value = 0

    print(f"\n=== Cross-correlation Analysis ===")
    print(f"Positive correlation: max = {positive_peak_value:.3f} at lag = {positive_peak_lag}")
    print(f"Negative correlation: min = {negative_peak_value:.3f} at lag = {negative_peak_lag}")
    print(f"================================\n")

    return ccf_values, lags, positive_peak_lag, positive_peak_value, negative_peak_lag, negative_peak_value, x_norm, y_norm


def create_ccf_plots(data1, data2, column_names, output_file=None):
    """
    Create cross-correlation plots

    Args:
        data1: First signal data
        data2: Second signal data
        column_names: Tuple of (col1_name, col2_name)
        output_file: Output file path (optional)
    """

    col1_name, col2_name = column_names

    # Create time index
    t = np.arange(len(data1))

    # Compute cross-correlation using normalized data
    ccf_values, lags, positive_peak_lag, positive_peak_value, negative_peak_lag, negative_peak_value, data1_norm, data2_norm = compute_cross_correlation(data1, data2)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Subplot 1: Time series comparison (using z-score normalized data)
    ax1.plot(t, data1_norm, label=f'{col1_name} (Signal A)', color='#1f77b4', linewidth=1.5)
    ax1.plot(t, data2_norm, label=f'{col2_name} (Signal B)', color='#d62728', linestyle='--', linewidth=1.5)
    ax1.set_title('Time Series Comparison (Z-score Normalized)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time Index', fontsize=12)
    ax1.set_ylabel('Z-score (σ)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Cross-correlation function
    ax2.plot(lags, ccf_values, color='#2ca02c', linewidth=2, label='Cross-correlation')

    # Add vertical lines for positive and negative peaks
    ax2.axvline(x=positive_peak_lag, color='#ff7f0e', linestyle=':', linewidth=2,
                label=f'Positive Peak Lag = {positive_peak_lag}')
    ax2.axvline(x=negative_peak_lag, color='#d62728', linestyle=':', linewidth=2,
                label=f'Negative Peak Lag = {negative_peak_lag}')

    ax2.set_title('Cross-correlation Function (CCF)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Lag (samples)', fontsize=12)
    ax2.set_ylabel('Correlation Coefficient', fontsize=12)

    # Focus on relevant lag range (adjust as needed)
    max_lag_display = min(50, len(lags) // 2)
    ax2.set_xlim([-max_lag_display, max_lag_display])

    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Add horizontal line at zero
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    # Add annotations for both peaks
    if abs(positive_peak_lag) <= max_lag_display:
        ax2.annotate(f'Positive Peak\n(Lag={positive_peak_lag}, Corr={positive_peak_value:.3f})',
                     xy=(positive_peak_lag, positive_peak_value),
                     xytext=(positive_peak_lag + max_lag_display * 0.1, positive_peak_value + 0.1),
                     arrowprops=dict(facecolor='orange', shrink=0.05, width=1, headwidth=5),
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    if abs(negative_peak_lag) <= max_lag_display:
        ax2.annotate(f'Negative Peak\n(Lag={negative_peak_lag}, Corr={negative_peak_value:.3f})',
                     xy=(negative_peak_lag, negative_peak_value),
                     xytext=(negative_peak_lag - max_lag_display * 0.3, negative_peak_value - 0.1),
                     arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=5),
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))

    # Add interpretation text
    if abs(positive_peak_value) > abs(negative_peak_value):
        if positive_peak_lag > 0:
            interpretation = f"{col1_name} leads {col2_name} by {positive_peak_lag} samples (positive correlation)"
        elif positive_peak_lag < 0:
            interpretation = f"{col2_name} leads {col1_name} by {abs(positive_peak_lag)} samples (positive correlation)"
        else:
            interpretation = f"Signals are synchronized (positive correlation: {positive_peak_value:.3f})"
    else:
        if negative_peak_lag > 0:
            interpretation = f"{col1_name} leads {col2_name} by {negative_peak_lag} samples (negative correlation)"
        elif negative_peak_lag < 0:
            interpretation = f"{col2_name} leads {col1_name} by {abs(negative_peak_lag)} samples (negative correlation)"
        else:
            interpretation = f"Signals are synchronized (negative correlation: {negative_peak_value:.3f})"

    fig.suptitle(f'Cross-Correlation Analysis: {col1_name} vs {col2_name}\n{interpretation}',
                 fontsize=16, fontweight='bold')

    # Adjust layout with increased spacing between title and plots
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.3)  # Increased spacing between title and top subplot
    
    # Save or show plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"CCF plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()
    
    return positive_peak_lag, positive_peak_value, negative_peak_lag, negative_peak_value


def main():
    """Main function"""
    # Configuration variables - modify these values as needed
    csv_file = '../data/processed/zh_dataset/0105/data_processed.csv'
    column1 = 'full_request_duration_ms_new_10.104.128.205:9093'
    column2 = 'nvidia_smi_ecc_errors_uncorrected_volatile_total_10.104.128.205:9835'
    output_file = '../data/processed/zh_dataset/0105/ccf.png'

    # Check if input file exists
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} does not exist")
        return

    print("=" * 60)
    print("Cross-Correlation Function Analysis from CSV")
    print("=" * 60)
    print(f"Input file: {csv_file}")
    print(f"Columns: {column1} vs {column2}")
    print("=" * 60)

    # Load data
    data1, data2, column_names = load_data_from_csv(csv_file, column1, column2)

    # Create plots
    positive_peak_lag, positive_peak_value, negative_peak_lag, negative_peak_value = create_ccf_plots(data1, data2, column_names, output_file)

    print("=" * 60)
    print("Cross-correlation analysis completed!")
    print(f"Positive correlation lag: {positive_peak_lag} samples")
    print(f"Positive correlation value: {positive_peak_value:.3f}")
    print(f"Negative correlation lag: {negative_peak_lag} samples")
    print(f"Negative correlation value: {negative_peak_value:.3f}")
    print(f"Output saved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()