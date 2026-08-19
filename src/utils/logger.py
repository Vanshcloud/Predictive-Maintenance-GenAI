"""
src/utils/logger.py — Centralized Logging Configuration
============================================================

WHY THIS FILE EXISTS:
    print() statements are the #1 sign of amateur code. They:
    - Can't be filtered by severity (debug vs error)
    - Can't be redirected to files
    - Can't be disabled in production
    - Don't include timestamps or context

    Loguru solves all of this with a beautiful, simple API.

WHY LOGURU OVER STDLIB LOGGING?
    Python's built-in `logging` module requires 10+ lines of
    boilerplate to configure. Loguru gives you:
    - Colored console output
    - Automatic timestamps
    - File rotation (5 MB max, keep 3 files)
    - Exception tracebacks with variable values
    - Zero configuration needed

USAGE:
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Processing started")
    logger.warning("Sensor reading out of range: {value}", value=105.3)
    logger.error("Model failed to load")

    # With context binding (adds equipment_id to every log line)
    eq_logger = logger.bind(equipment_id="PUMP-001")
    eq_logger.info("Prediction complete")
    # Output: 2024-01-15 10:30:45 | INFO | equipment_id=PUMP-001 | Prediction complete
"""

import sys
from pathlib import Path

from loguru import logger

from config.settings import PROJECT_ROOT, get_settings


def setup_logger() -> None:
    """
    Configure the application-wide logger.

    This function is called ONCE at application startup.
    It sets up:
    1. Console output — colored, human-readable (for development)
    2. File output — structured, rotated (for production debugging)
    """
    settings = get_settings()

    # ------------------------------------------------------------------
    # Remove the default handler
    # ------------------------------------------------------------------
    # WHY: Loguru ships with a default stderr handler. We remove it
    # and add our own with custom formatting.
    # ------------------------------------------------------------------
    logger.remove()

    # ------------------------------------------------------------------
    # Console Handler — for development
    # ------------------------------------------------------------------
    # FORMAT EXPLANATION:
    # {time:HH:mm:ss}  — Timestamp (just time, not date, for readability)
    # {level: <8}       — Log level, left-padded to 8 chars for alignment
    # {name}            — Module name (e.g., src.data.ingestion)
    # {message}         — The actual log message
    # ------------------------------------------------------------------
    console_format = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=console_format,
        level=settings.LOG_LEVEL,
        colorize=True,
    )

    # ------------------------------------------------------------------
    # File Handler — for production debugging
    # ------------------------------------------------------------------
    # WHY rotation="5 MB"?
    #   Log files can grow to gigabytes. Rotation prevents disk exhaustion.
    #   5 MB is enough to capture recent history without wasting space.
    #
    # WHY retention="3 days"?
    #   Keeps 3 days of logs for debugging, then auto-deletes older files.
    # ------------------------------------------------------------------
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )

    logger.add(
        str(log_dir / "app_{time:YYYY-MM-DD}.log"),
        format=file_format,
        level="DEBUG",
        rotation="5 MB",
        retention="3 days",
        compression="zip",  # Compress old log files to save space
        enqueue=True,  # Thread-safe logging (important for FastAPI)
    )


def get_logger(name: str = __name__) -> logger:
    """
    Get a logger instance bound with the module name.

    WHY a factory function?
        So each module gets a logger with its name attached,
        making it easy to trace which module generated a log entry.

    Args:
        name: Module name (typically __name__). This appears in log
              output so you know which file generated the message.

    Returns:
        A loguru logger instance bound with the module name.

    Example:
        logger = get_logger(__name__)
        logger.info("Sensor data loaded successfully")
    """
    return logger.bind(module=name)


# ------------------------------------------------------------------
# Auto-setup on first import
# ------------------------------------------------------------------
# WHY: We want logging to be configured as soon as any module
# imports the logger. This ensures no log messages are lost.
# ------------------------------------------------------------------
setup_logger()
