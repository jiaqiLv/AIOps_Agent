"""Time utilities for time window parsing

Convention: all Unix timestamps in this system are "naive Beijing timestamps" —
the number of seconds from 1970-01-01 00:00:00 UTC to the Beijing-time moment,
treating Beijing time as if it were UTC (no +8h offset).

Display: convert back via UTC to recover the original Beijing time string.
"""

import re
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Union


def parse_time_range(time_str: str) -> Optional[Dict[str, str]]:
    """
    Parse time range string like "2025-04-06 10:00 到 2025-04-06 11:00".

    Args:
        time_str: Time range string

    Returns:
        Dictionary with start_time and end_time, or None if parsing fails
    """
    # Pattern: YYYY-MM-DD HH:MM 到 YYYY-MM-DD HH:MM
    pattern = r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s*[到至]\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})'
    match = re.search(pattern, time_str)

    if match:
        return {
            "start_time": match.group(1),
            "end_time": match.group(2)
        }
    return None


def format_timestamp(dt: datetime) -> str:
    """
    Format datetime to string.

    Args:
        dt: Datetime object

    Returns:
        Formatted timestamp string
    """
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def parse_timestamp(ts_str: str) -> Optional[datetime]:
    """
    Parse timestamp string to datetime.

    Args:
        ts_str: Timestamp string

    Returns:
        Datetime object or None if parsing fails
    """
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue

    return None


def format_unix_ts(ts: Union[float, int, None]) -> str:
    """Format a naive Beijing Unix timestamp as Beijing time string.

    Naive Beijing timestamps treat the Beijing time moment as UTC,
    so converting back via UTC recovers the original Beijing time string.

    Returns:
        "YYYY-MM-DD HH:MM:SS" in Beijing time, or "N/A" if ts is None.
    """
    if ts is None:
        return "N/A"
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except (OSError, ValueError, OverflowError):
        return str(ts)


def format_inject_time(ts: Union[float, int, None]) -> str:
    """Format inject_time with both human-readable time and Unix timestamp.

    Convenience wrapper around format_unix_ts that appends the raw Unix value.
    """
    if ts is None:
        return "未提供"
    dt_str = format_unix_ts(ts)
    return f"{dt_str} (Unix: {ts})"
