import logging
import sys
from .config import Config

def setup_logging(name=None):
    """
    Sets up logging with a consistent format.
    Use this at the start of scripts.
    """
    logger = logging.getLogger(name)
    logger.setLevel(Config.LOG_LEVEL)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger
