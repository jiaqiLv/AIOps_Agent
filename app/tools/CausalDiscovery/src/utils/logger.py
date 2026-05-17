"""
Logger setup utility.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(name: Optional[str] = None,
                level: int = logging.INFO,
                log_file: Optional[str] = None,
                format_string: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with consistent formatting.

    Also sets the root logger level to ensure all loggers output at the specified level.

    Args:
        name: Logger name (uses __name__ if None)
        level: Logging level
        log_file: Optional file path to log to
        format_string: Custom format string

    Returns:
        Configured logger instance
    """
    # Set root logger level to ensure all loggers output at the specified level
    logging.getLogger().setLevel(level)

    if name is None:
        name = __name__

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
