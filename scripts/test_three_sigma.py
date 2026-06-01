"""Test script for 3-sigma anomaly detection tool.

Usage:
    python scripts/test_three_sigma.py <csv_path> <inject_time> [options]

Examples:
    python scripts/test_three_sigma.py data/sample_metrics.csv 50 --time-column timestamp
    python scripts/test_three_sigma.py data/ZH_dataset/0105/data.csv "2026-01-05 05:48:00"
"""

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta

import pandas as pd

# Add project root to path
sys.path.insert(0, ".")

from app.tools.three_sigma import run_three_sigma


def parse_inject_time(value: str) -> float:
    """Parse inject_time from string to Unix timestamp."""
    try:
        return float(value)
    except ValueError:
        pass

    # Try common datetime formats
    for fmt in [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
    ]:
        try:
            dt = datetime.strptime(value, fmt)
            dt = dt.replace(tzinfo=timezone(timedelta(hours=8)))
            return dt.timestamp()
        except ValueError:
            continue

    print(f"Error: Cannot parse inject_time '{value}'")
    print("Supported formats: Unix timestamp, '2026-01-05 05:48:00', '2026/01/05 05:48:00'")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Test 3-sigma anomaly detection")
    parser.add_argument("--csv_path", help="Path to CSV data file", default='../data/ZH_dataset/0105/data.csv')
    parser.add_argument("--inject_time", help="Fault injection time (Unix timestamp or datetime string)", default='2026-01-05 05:48:00')
    parser.add_argument("--time-column", default="time", help="Time column name (default: time)")
    parser.add_argument("--baseline-start", type=int, default=30,
                        help="Minutes before inject_time where baseline window starts (default: 30)")
    parser.add_argument("--baseline-end", type=int, default=60,
                        help="Minutes before inject_time where baseline window ends (default: 60)")
    parser.add_argument("--detect-before", type=int, default=10,
                        help="Minutes before inject_time to include in detection (default: 10)")
    parser.add_argument("--detect-after", type=int, default=10,
                        help="Minutes after inject_time for detection (default: 10)")
    parser.add_argument("--threshold", type=float, default=3.0,
                        help="Z-score threshold (default: 3.0)")
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="Specific metric columns to check (default: all numeric)")

    args = parser.parse_args()

    # Load data
    print(f"Loading data from: {args.csv_path}")
    try:
        df = pd.read_csv(args.csv_path)
    except FileNotFoundError:
        print(f"Error: File not found: {args.csv_path}")
        sys.exit(1)

    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Time column: '{args.time_column}'")
    print()

    if args.time_column not in df.columns:
        print(f"Error: Time column '{args.time_column}' not found.")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)

    # Parse inject_time
    inject_time = parse_inject_time(args.inject_time)

    tz = timezone(timedelta(hours=8))
    inject_dt = datetime.fromtimestamp(inject_time, tz=tz)
    print(f"Inject time: {inject_dt.strftime('%Y-%m-%d %H:%M:%S')} (Unix: {inject_time})")
    print()

    # Print window info
    bl_start_dt = datetime.fromtimestamp(inject_time - args.baseline_end * 60, tz=tz)
    bl_end_dt = datetime.fromtimestamp(inject_time - args.baseline_start * 60, tz=tz)
    det_start_dt = datetime.fromtimestamp(inject_time - args.detect_before * 60, tz=tz)
    det_end_dt = datetime.fromtimestamp(inject_time + args.detect_after * 60, tz=tz)

    print(f"Baseline window: [{bl_start_dt.strftime('%H:%M:%S')}, {bl_end_dt.strftime('%H:%M:%S')})")
    print(f"Detection window: [{det_start_dt.strftime('%H:%M:%S')}, {det_end_dt.strftime('%H:%M:%S')}]")
    print(f"Threshold: {args.threshold}")
    print()

    # Run detection
    result_json = run_three_sigma(
        data=df,
        inject_time=inject_time,
        baseline_start_minutes=args.baseline_start,
        baseline_end_minutes=args.baseline_end,
        detect_before_minutes=args.detect_before,
        detect_minutes=args.detect_after,
        threshold=args.threshold,
        metric_columns=args.metrics,
        time_column=args.time_column,
    )

    result = json.loads(result_json)

    # Print results
    print("=" * 60)
    if not result.get("success"):
        print(f"Detection FAILED: {result.get('error', 'Unknown error')}")
        sys.exit(1)

    print(f"Algorithm: {result['algorithm']}")
    print(f"Metrics checked: {result['metrics_checked']}")
    print(f"Anomalous metrics: {result['anomalous_metric_count']}")
    print(f"Total anomaly points: {result['anomalies_found']}")
    print(f"Baseline data points: {result['baseline_points']}")
    print(f"Detection data points: {result['detection_points']}")
    print()

    # Print per-metric summary
    anomalies_by_metric = result.get("anomalies_by_metric", {})
    if not anomalies_by_metric:
        print("No anomalies detected.")
        return

    print("-" * 60)
    print("Anomaly Details (sorted by max z-score)")
    print("-" * 60)

    for i, (metric, info) in enumerate(anomalies_by_metric.items(), 1):
        atype = info["anomaly_type"]
        type_cn = "突增 (sudden_increase)" if atype == "sudden_increase" else "骤降 (sudden_decrease)"
        max_z = info["max_z_score"]
        points = info["points"]

        print(f"\n[{i}] {metric}")
        print(f"    Anomaly type: {type_cn}")
        print(f"    Max z-score:  {max_z:.4f}")
        print(f"    Points:       {len(points)}")

        # Time range
        timestamps = [p["timestamp"] for p in points]
        t_min = datetime.fromtimestamp(min(timestamps), tz=tz)
        t_max = datetime.fromtimestamp(max(timestamps), tz=tz)
        print(f"    Time range:   {t_min.strftime('%H:%M:%S')} ~ {t_max.strftime('%H:%M:%S')}")

        # Value range
        values = [p["value"] for p in points]
        z_scores = [p["z_score"] for p in points]
        print(f"    Value range:  {min(values):.4f} ~ {max(values):.4f}")
        print(f"    Z-score range: {min(z_scores):.4f} ~ {max(z_scores):.4f}")

        # Print each anomaly point
        print(f"    Anomaly points:")
        for p in points:
            pt_dt = datetime.fromtimestamp(p["timestamp"], tz=tz)
            print(f"      row={p['index']:>4d}  "
                  f"time={pt_dt.strftime('%H:%M:%S')}  "
                  f"value={p['value']:>10.4f}  "
                  f"z={p['z_score']:.4f}")

    print()
    print("=" * 60)

    # Print raw JSON (compact)
    print("\nRaw JSON output:")
    print(json.dumps(result, indent=2, ensure_ascii=False)[:3000])
    if len(json.dumps(result)) > 3000:
        print("... (truncated)")


if __name__ == "__main__":
    main()
