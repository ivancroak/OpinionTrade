"""
Logging utilities for Opinion Trade Bot.

This module sets up logging to both console and file.
Logs are saved to the logs/ folder with daily rotation.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

# Try to import colorama for colored console output
try:
    from colorama import init, Fore, Style
    init(autoreset=True)  # Auto-reset colors after each print
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to log levels in console output.
    """

    # Color mapping for different log levels
    COLORS = {
        'DEBUG': Fore.CYAN if COLORAMA_AVAILABLE else '',
        'INFO': Fore.GREEN if COLORAMA_AVAILABLE else '',
        'WARNING': Fore.YELLOW if COLORAMA_AVAILABLE else '',
        'ERROR': Fore.RED if COLORAMA_AVAILABLE else '',
        'CRITICAL': Fore.RED + Style.BRIGHT if COLORAMA_AVAILABLE else '',
    }
    RESET = Style.RESET_ALL if COLORAMA_AVAILABLE else ''

    def format(self, record):
        # Add color to the level name
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


# Global logger instance
_logger: Optional[logging.Logger] = None


def setup_logger(
    level: str = "INFO",
    log_to_file: bool = True,
    log_file_prefix: str = "bot",
    logs_dir: str = "logs"
) -> logging.Logger:
    """
    Set up the global logger with console and optional file output.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to also save logs to a file
        log_file_prefix: Prefix for log file names (e.g., "bot" -> "bot_2024-01-15.log")
        logs_dir: Directory to store log files

    Returns:
        Configured logger instance
    """
    global _logger

    # Create logger
    _logger = logging.getLogger("opinion_bot")
    _logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove any existing handlers (in case of re-init)
    _logger.handlers.clear()

    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_format = ColoredFormatter(
        fmt="%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    _logger.addHandler(console_handler)

    # File handler (if enabled)
    if log_to_file:
        # Create logs directory if it doesn't exist
        logs_path = Path(logs_dir)
        logs_path.mkdir(exist_ok=True)

        # Create log file with today's date
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = logs_path / f"{log_file_prefix}_{today}.log"

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            fmt="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        _logger.addHandler(file_handler)

        _logger.debug(f"Logging to file: {log_file}")

    return _logger


def get_logger() -> logging.Logger:
    """
    Get the global logger instance.
    If not set up yet, creates a default one.

    Returns:
        Logger instance
    """
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger


# Convenience functions for quick logging
def log_info(message: str):
    """Log an info message."""
    get_logger().info(message)


def log_debug(message: str):
    """Log a debug message."""
    get_logger().debug(message)


def log_warning(message: str):
    """Log a warning message."""
    get_logger().warning(message)


def log_error(message: str):
    """Log an error message."""
    get_logger().error(message)


def log_trade(action: str, details: str):
    """
    Log a trade-related action with special formatting.

    Args:
        action: What happened (e.g., "ORDER PLACED", "ORDER FILLED", "ORDER CANCELLED")
        details: Additional details
    """
    get_logger().info(f"[TRADE] {action}: {details}")


def log_wallet(wallet_id: int, message: str):
    """
    Log a message for a specific wallet.

    Args:
        wallet_id: The wallet number
        message: The message to log
    """
    get_logger().info(f"[Wallet #{wallet_id}] {message}")
