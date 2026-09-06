# Absolute Path: <project_root>/core/logger.py

"""
logger.py — Centralized, thread-safe structured logging for I-POMCP.

Provides process-safe structured logging with PID attribution for distributed rollouts
and multi-threaded MCTS simulations. Supports both console streams and artifact files.
"""

import logging
import sys
import os


def get_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Retrieves or creates a configured logger instance.
    Safe to call across multiprocessing forks and spawns.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | PID:%(process)-5d | %(name)-18s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Attach Console Handler if not already present
    has_stream_handler = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_stream_handler:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Attach File Handler if requested and not already attached for this specific path
    if log_file:
        norm_path = os.path.abspath(log_file)
        has_file_handler = any(
            isinstance(h, logging.FileHandler) and os.path.abspath(getattr(h, "baseFilename", "")) == norm_path
            for h in logger.handlers
        )
        if not has_file_handler:
            os.makedirs(os.path.dirname(norm_path), exist_ok=True)
            file_handler = logging.FileHandler(norm_path, mode="a")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger