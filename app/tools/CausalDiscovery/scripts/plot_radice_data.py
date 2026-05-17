"""
Plot RADICE Data - Line Chart Visualization

This script reads a RADICE data.csv file and creates a vertical stack of line charts,
one for each metric (column) in the dataset.

Usage:
    python plot_radice_data.py --data data/raw/RADICE/N15/artificialResults_0/data.csv
    python plot_radice_data.py --data data/raw/RADICE/N15/artificialResults_0/data.csv --output output.png
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# Set matplotlib to use a non-interactive backend for server environments
matplotlib.use('Agg')


def plot_radice_metrics(data_path: str, output_path: str = None, figsize: tuple = (12, 20)):
    """
    Plot all metrics from a RADICE data.csv file as vertically stacked line charts.

    Args:
        data_path: Path to the data.csv file
        output_path: Path to save the output figure (optional)
        figsize: Figure size (width, height) in inches
    """
    # Read data
    df = pd.read_csv(data_path)

    print(f"Loaded data: {df.shape[0]} samples, {df.shape[1]} metrics")
    print(f"Metrics: {list(df.columns)}")

    # Create subplots - one for each metric
    n_metrics = df.shape[1]
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)

    # Handle single metric case
    if n_metrics == 1:
        axes = [axes]

    # Plot each metric
    for i, col in enumerate(df.columns):
        ax = axes[i]
        ax.plot(df.index, df[col], linewidth=1, color='steelblue')
        ax.set_ylabel(str(col), fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=8)

    # Set common x-label
    axes[-1].set_xlabel('Time Step', fontsize=10)

    plt.suptitle(f'RADICE Metrics: {Path(data_path).parent.name}',
                 fontsize=14, y=0.999)
    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    else:
        # Default output path
        default_output = Path(data_path).parent / 'metrics_plot.png'
        plt.savefig(default_output, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {default_output}")

    plt.close()


def plot_radice_metrics_compact(data_path: str, output_path: str = None,
                                figsize: tuple = (14, 10)):
    """
    Plot all metrics in a more compact grid layout.

    Args:
        data_path: Path to the data.csv file
        output_path: Path to save the output figure (optional)
        figsize: Figure size (width, height) in inches
    """
    # Read data
    df = pd.read_csv(data_path)

    n_metrics = df.shape[1]

    # Calculate grid dimensions (aim for roughly 3-4 columns)
    n_cols = min(4, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_metrics > 1 else [axes]

    # Plot each metric
    for i, col in enumerate(df.columns):
        ax = axes[i]
        ax.plot(df.index, df[col], linewidth=0.8, color='steelblue')
        ax.set_title(f'Metric {col}', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=7)

    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f'RADICE Metrics: {Path(data_path).parent.name}',
                 fontsize=14, y=0.995)
    plt.tight_layout()

    # Save
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved compact plot to: {output_path}")
    else:
        default_output = Path(data_path).parent / 'metrics_plot_compact.png'
        plt.savefig(default_output, dpi=150, bbox_inches='tight')
        print(f"Saved compact plot to: {default_output}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot RADICE data metrics as line charts'
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Path to the data.csv file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output path for the plot (default: data.csv parent directory)'
    )
    parser.add_argument(
        '--layout', '-l',
        type=str,
        choices=['vertical', 'compact'],
        default='vertical',
        help='Layout style: vertical (stacked) or compact (grid)'
    )
    parser.add_argument(
        '--width', '-w',
        type=float,
        default=12,
        help='Figure width in inches (default: 12)'
    )
    parser.add_argument(
        '--height',
        type=float,
        default=None,
        help='Figure height in inches (default: auto-calculated)'
    )

    args = parser.parse_args()

    # Calculate height if not specified
    if args.height is None:
        n_metrics = len(pd.read_csv(args.data).columns)
        if args.layout == 'vertical':
            args.height = max(10, n_metrics * 1.2)
        else:
            args.height = 10

    figsize = (args.width, args.height)

    # Plot based on layout choice
    if args.layout == 'vertical':
        plot_radice_metrics(args.data, args.output, figsize)
    else:
        plot_radice_metrics_compact(args.data, args.output, figsize)


if __name__ == '__main__':
    main()