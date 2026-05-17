"""Path resolution utilities for data files"""

import os
from typing import Optional
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Cache the project root to avoid repeated calculations
_project_root = None


def get_project_root() -> str:
    """
    Get the project root directory.

    The project root is the parent directory of the 'app' folder.

    Returns:
        Absolute path to the project root directory
    """
    global _project_root
    if _project_root is None:
        # Get the directory containing this file (app/utils/)
        # Go up two levels to get to the project root
        _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _project_root


def resolve_config_path(config_path: str) -> str:
    """
    Resolve a config file path relative to the project root.

    Args:
        config_path: Path to config file (can be relative or absolute)

    Returns:
        Absolute path to the config file
    """
    if os.path.isabs(config_path):
        return config_path

    # Resolve relative to project root
    project_root = get_project_root()
    full_path = os.path.join(project_root, config_path)

    if os.path.exists(full_path):
        return full_path

    # If not found, return the full path anyway (will error later with better message)
    logger.warning(f"Config path does not exist: {full_path}")
    return full_path


def resolve_data_path(
    data_path: Optional[str] = None,
    benchmark: Optional[str] = None,
    instance: Optional[str] = None,
    case: Optional[str] = None,
    data_root: str = "data"
) -> str:
    """
    Resolve the full data path based on various input formats.

    Args:
        data_path: Direct path to data file
        benchmark: Dataset name (e.g., "RE1-OB")
        instance: Instance name (e.g., "adservice_cpu")
        case: Case identifier (e.g., "case_001" or "1")
        data_root: Root directory for data (default: "data")

    Returns:
        Full path to the data CSV file

    Raises:
        ValueError: If path cannot be resolved
    """
    # If direct path is provided, use it
    if data_path:
        if os.path.isabs(data_path):
            return data_path

        # Resolve relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        full_path = os.path.join(project_root, data_path)

        if os.path.exists(full_path):
            return full_path

        logger.warning(f"Data path does not exist: {full_path}")

        # Try case-insensitive search
        if not os.path.exists(full_path):
            dir_name = os.path.dirname(full_path)
            file_name = os.path.basename(full_path)

            if os.path.exists(dir_name):
                for f in os.listdir(dir_name):
                    if f.lower() == file_name.lower():
                        return os.path.join(dir_name, f)

        return full_path

    # Build path from components
    if benchmark and instance and case:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # Normalize case identifier
        case_dir = case
        if case.isdigit():
            case_dir = f"case_{case.zfill(3)}"
        elif not case.startswith("case_"):
            case_dir = f"case_{case}"

        # Build path
        data_path = os.path.join(data_root, benchmark, instance, case_dir, "data.csv")
        full_path = os.path.join(project_root, data_path)

        if os.path.exists(full_path):
            return full_path

        # Try without case_ prefix
        data_path = os.path.join(data_root, benchmark, instance, case, "data.csv")
        full_path = os.path.join(project_root, data_path)

        if os.path.exists(full_path):
            return full_path

        logger.warning(f"Data path does not exist: {full_path}")
        return full_path

    raise ValueError(
        "Cannot resolve data path. Provide either:\n"
        "- data_path: Direct path to CSV file\n"
        "- benchmark + instance + case: Components to build path"
    )


def find_data_files(
    benchmark: Optional[str] = None,
    instance: Optional[str] = None,
    data_root: str = "data"
) -> list:
    """
    Find all available data files matching the criteria.

    Args:
        benchmark: Dataset name filter (optional)
        instance: Instance name filter (optional)
        data_root: Root directory for data

    Returns:
        List of available data file paths
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    search_root = os.path.join(project_root, data_root)

    if not os.path.exists(search_root):
        logger.warning(f"Data root does not exist: {search_root}")
        return []

    results = []

    # Walk through directory tree
    for root, dirs, files in os.walk(search_root):
        # Check if this directory matches our filters
        rel_path = os.path.relpath(root, search_root)
        parts = rel_path.split(os.sep)

        if len(parts) < 2:
            continue

        path_benchmark = parts[0] if len(parts) > 0 else None
        path_instance = parts[1] if len(parts) > 1 else None

        # Apply filters
        if benchmark and path_benchmark != benchmark:
            continue
        if instance and path_instance != instance:
            continue

        # Look for data.csv files
        if "data.csv" in files:
            results.append(os.path.join(root, "data.csv"))

    return sorted(results)


def validate_data_path(data_path: str) -> bool:
    """
    Validate that a data file exists and is readable.

    Args:
        data_path: Path to the data file

    Returns:
        True if file exists and is readable
    """
    if not os.path.exists(data_path):
        return False

    if not os.path.isfile(data_path):
        return False

    # Check if file is readable
    try:
        with open(data_path, 'r') as f:
            f.read(1)
        return True
    except Exception:
        return False
