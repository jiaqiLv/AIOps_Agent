"""CSV reader tool for reading metrics data"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
from app.utils.logger import get_logger
from app.utils.path_resolver import resolve_data_path

logger = get_logger(__name__)


def read_csv_headers(
    file_path: str,
    preview_rows: int = 5,
    encoding: str = "utf-8"
) -> Dict[str, Any]:
    """
    Read CSV file headers and preview data.

    Args:
        file_path: Path to CSV file
        preview_rows: Number of rows to preview
        encoding: File encoding

    Returns:
        Dictionary containing headers, shape, preview data, and status
    """
    try:
        # Validate file path
        path = Path(file_path)
        if not path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}"
            }

        # Read CSV
        df = pd.read_csv(file_path, encoding=encoding)

        # Remove duplicate columns (keep first occurrence)
        dup_cols = df.columns[df.columns.duplicated()].unique().tolist()
        if dup_cols:
            logger.warning(f"Removing {len(dup_cols)} duplicate column(s): {dup_cols}")
            df = df.loc[:, ~df.columns.duplicated()]

        # Prepare result
        result = {
            "success": True,
            "headers": df.columns.tolist(),
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "preview": df.head(preview_rows).to_dict("records"),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "file_path": str(path.absolute())
        }

        logger.info(f"Successfully read CSV: {file_path}, shape: {result['shape']}")
        return result

    except pd.errors.EmptyDataError:
        return {
            "success": False,
            "error": "CSV file is empty"
        }
    except UnicodeDecodeError:
        return {
            "success": False,
            "error": f"Encoding error, try different encoding (current: {encoding})"
        }
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def read_csv_data(
    file_path: str,
    columns: Optional[list] = None,
    filters: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Read CSV data with optional column selection and filtering.

    Args:
        file_path: Path to CSV file
        columns: List of columns to read (None for all)
        filters: Dictionary of column:value pairs for filtering
        limit: Maximum number of rows to return

    Returns:
        Dictionary containing data and status
    """
    try:
        # Read CSV
        df = pd.read_csv(file_path)

        # Remove duplicate columns (keep first occurrence)
        dup_cols = df.columns[df.columns.duplicated()].unique().tolist()
        if dup_cols:
            logger.warning(f"Removing {len(dup_cols)} duplicate column(s): {dup_cols}")
            df = df.loc[:, ~df.columns.duplicated()]

        # Select columns
        if columns:
            available_cols = [c for c in columns if c in df.columns]
            if available_cols:
                df = df[available_cols]

        # Apply filters
        if filters:
            for col, value in filters.items():
                if col in df.columns:
                    df = df[df[col] == value]

        # Limit rows
        if limit and len(df) > limit:
            df = df.head(limit)

        return {
            "success": True,
            "data": df.to_dict("records"),
            "rows": len(df),
            "columns": df.columns.tolist()
        }

    except Exception as e:
        logger.error(f"Error reading CSV data: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def read_csv_metrics(
    data_path: str,
    benchmark: Optional[str] = None,
    instance: Optional[str] = None,
    case: Optional[str] = None
) -> Dict[str, Any]:
    """
    Read CSV metrics data for root cause analysis.

    This is the main CSV reader function that conforms to the
    system design document specification.

    Args:
        data_path: Direct path to CSV file
        benchmark: Dataset name (alternative to data_path)
        instance: Instance name (alternative to data_path)
        case: Case identifier (alternative to data_path)

    Returns:
        Dictionary containing:
        - success: bool
        - columns: list of column names
        - shape: tuple of (rows, columns)
        - preview: list of preview rows
        - dtypes: dict of column data types
        - data_path: resolved data path
        - error: str or None
    """
    try:
        # Resolve data path
        resolved_path = resolve_data_path(
            data_path=data_path,
            benchmark=benchmark,
            instance=instance,
            case=case
        )

        # Read CSV
        df = pd.read_csv(resolved_path)

        # Remove duplicate columns (keep first occurrence)
        dup_cols = df.columns[df.columns.duplicated()].unique().tolist()
        if dup_cols:
            logger.warning(f"Removing {len(dup_cols)} duplicate column(s): {dup_cols}")
            df = df.loc[:, ~df.columns.duplicated()]

        # Extract metric columns (exclude non-numeric columns like 'time')
        metric_columns = df.select_dtypes(include=['number']).columns.tolist()

        result = {
            "success": True,
            "columns": df.columns.tolist(),
            "metric_columns": metric_columns,
            "shape": (len(df), len(df.columns)),
            "preview": df.head(5).to_dict("records"),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "data_path": resolved_path,
            "num_rows": len(df),
            "num_metrics": len(metric_columns)
        }

        logger.info(f"Successfully read CSV: {resolved_path}, shape: {result['shape']}")
        return result

    except FileNotFoundError as e:
        return {
            "success": False,
            "error": f"File not found: {str(e)}",
            "data_path": data_path
        }
    except pd.errors.EmptyDataError:
        return {
            "success": False,
            "error": "CSV file is empty",
            "data_path": data_path
        }
    except Exception as e:
        logger.error(f"Error reading CSV metrics: {e}")
        return {
            "success": False,
            "error": str(e),
            "data_path": data_path
        }
