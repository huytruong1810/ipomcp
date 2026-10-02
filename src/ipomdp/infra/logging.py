"""
logger.py — Centralized, thread-safe structured logging for I-POMCP.

Provides standard Python logging with PID attribution. File handlers belong to
the configuring process; multiple processes sharing a file are not guaranteed
atomic records. Batch data integrity comes from per-trial atomic checkpoints.
"""

import logging
import os
import sys


def get_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Retrieves or creates a configured logger instance.
    Safe to call across multiprocessing forks and spawns.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | PID:%(process)-5d | %(name)-18s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
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
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler) and handler.baseFilename != norm_path:
                logger.removeHandler(handler)
                handler.close()
        has_file_handler = any(
            isinstance(h, logging.FileHandler)
            and os.path.abspath(getattr(h, "baseFilename", "")) == norm_path
            for h in logger.handlers
        )
        if not has_file_handler:
            os.makedirs(os.path.dirname(norm_path), exist_ok=True)
            file_handler = logging.FileHandler(norm_path, mode="a")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
