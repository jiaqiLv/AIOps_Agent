"""Validation utilities"""

import re
from pathlib import Path
from typing import Optional


def validate_csv_path(file_path: str) -> bool:
    """
    Validate if a CSV file path exists and is valid.

    Args:
        file_path: Path to CSV file

    Returns:
        True if valid, False otherwise
    """
    path = Path(file_path)
    return path.exists() and path.is_file() and path.suffix.lower() == ".csv"


def validate_service_name(service_name: str) -> bool:
    """
    Validate service name format.

    Args:
        service_name: Service name to validate

    Returns:
        True if valid format, False otherwise
    """
    # Service names can contain letters, numbers, hyphens, underscores
    pattern = r'^[a-zA-Z0-9_-]+$'
    return bool(re.match(pattern, service_name))


def validate_metric_name(metric_name: str) -> bool:
    """
    Validate metric name format.

    Args:
        metric_name: Metric name to validate

    Returns:
        True if valid format, False otherwise
    """
    # Metric names can contain letters, numbers, underscores, dots
    pattern = r'^[a-zA-Z0-9_.]+$'
    return bool(re.match(pattern, metric_name))


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file operations.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    return re.sub(r'[<>:"/\\|?*]', '_', filename)
