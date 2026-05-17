"""JSON utilities for data handling and serializing complex objects"""

import json
from typing import Any, Dict, Optional, List
from datetime import datetime
import pandas as pd
import numpy as np
from app.utils.logger import get_logger

logger = get_logger(__name__)


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles pandas, numpy, and datetime objects"""

    def default(self, obj: Any) -> Any:
        # Handle pandas objects
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()

        # Handle numpy objects
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # Handle datetime
        if isinstance(obj, datetime):
            return obj.isoformat()

        # Handle sets
        if isinstance(obj, set):
            return list(obj)

        # Let the base class handle it or raise TypeError
        return super().default(obj)


def pretty_print_json(data: Dict[str, Any], indent: int = 2) -> str:
    """
    Pretty print JSON data.

    Args:
        data: Dictionary to pretty print
        indent: Indentation spaces

    Returns:
        Formatted JSON string
    """
    return json.dumps(data, indent=indent, ensure_ascii=False, cls=JSONEncoder)


def safe_json_loads(json_str: str, default: Optional[Any] = None) -> Any:
    """
    Safely load JSON string.

    Args:
        json_str: JSON string to parse
        default: Default value if parsing fails

    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def to_json(obj: Any, indent: int = 2, ensure_ascii: bool = False) -> str:
    """
    Convert an object to JSON string.

    Args:
        obj: The object to serialize
        indent: Number of spaces for indentation
        ensure_ascii: Whether to escape non-ASCII characters

    Returns:
        JSON string representation
    """
    return json.dumps(obj, cls=JSONEncoder, indent=indent, ensure_ascii=ensure_ascii)


def to_json_file(obj: Any, filepath: str, indent: int = 2, ensure_ascii: bool = False) -> None:
    """
    Write an object to a JSON file.

    Args:
        obj: The object to serialize
        filepath: Path to the output file
        indent: Number of spaces for indentation
        ensure_ascii: Whether to escape non-ASCII characters
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(obj, f, cls=JSONEncoder, indent=indent, ensure_ascii=ensure_ascii)

    logger.debug(f"Wrote JSON to {filepath}")


def from_json_file(filepath: str) -> Any:
    """
    Read and parse a JSON file.

    Args:
        filepath: Path to the JSON file

    Returns:
        Parsed object
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def sanitize_for_json(obj: Any) -> Any:
    """
    Sanitize an object for JSON serialization by converting non-serializable types.

    Args:
        obj: The object to sanitize

    Returns:
        A JSON-serializable version of the object
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.to_dict()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return obj


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract JSON from a text that may contain other content.

    Args:
        text: Text that may contain JSON

    Returns:
        Extracted and parsed JSON object

    Raises:
        ValueError: If no valid JSON is found
    """
    import re

    # Try to find JSON object
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Try to find JSON array
    array_match = re.search(r'\[.*\]', text, re.DOTALL)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError("No valid JSON found in text")
