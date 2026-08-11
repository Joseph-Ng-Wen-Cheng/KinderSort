"""
utils.py — Utility functions for KinderSort logging and file operations.
"""

import logging
from pathlib import Path


def setup_logger(output_dir: Path) -> logging.Logger:
    """Set up logging to file in output directory.
    
    Args:
        output_dir: Path to output directory where log will be written
        
    Returns:
        Configured logger instance
    """
    log_file = output_dir / "kindersort_log.txt"
    
    logger = logging.getLogger("KinderSort")
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # File handler (detailed logging)
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (less verbose)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger
