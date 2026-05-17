"""Time utilities for time window parsing"""

import re
from datetime import datetime
from typing import Optional, Dict, Any


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
