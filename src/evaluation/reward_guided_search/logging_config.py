# logging_config.py

import logging
from typing import Optional

import colorlog
import os


def setup_logger(name: Optional[str] = None) -> logging.Logger:
    fmt_string = '%(log_color)s %(asctime)s - %(levelname)s - %(message)s'
    log_colors = {
        'DEBUG': 'white',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'purple'
    }
    
    rank = int(os.environ.get("RANK", 0))
    logger = colorlog.getLogger(name)
    
    if rank == 0:
        if not logger.handlers:
            handler = colorlog.StreamHandler()
            formatter = colorlog.ColoredFormatter(fmt_string, log_colors=log_colors)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(colorlog.INFO)
    else:
        logger.setLevel(logging.CRITICAL + 1)  # Disable logging for other ranks

    return logger

# Root logger to be used throughout the project
logger = setup_logger()