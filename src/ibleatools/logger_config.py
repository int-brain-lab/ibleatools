from iblutil.util import setup_logger as ibl_setup_logger
from pathlib import Path
import logging

def setup_logger(name: str, log_level: int = logging.INFO, log_path: str = None) -> logging.Logger:
    """
    Set up a logger using iblutil.util.setup_logger with configurable log path.
    
    Args:
        name: Name of the logger
        log_level: Logging level (default: INFO)
        log_path: Optional absolute path for log file. If None, no file logging will be done.
    
    Returns:
        Configured logger instance
    """
    if log_path:
        # Ensure the log path is absolute
        log_path = Path(log_path).resolve()
        # Create parent directories if they don't exist
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return ibl_setup_logger(name=name, level=log_level, file=str(log_path))
    else:
        return ibl_setup_logger(name=name, level=log_level) 