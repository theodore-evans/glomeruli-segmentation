import logging
import sys

FORMATTER = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

def get_log_level(verbosity: int, default_log_level=logging.WARN, min_log_level=logging.DEBUG):
    return max(default_log_level - 10 * verbosity, min_log_level)

def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_logger(logger_name, log_level):
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.addHandler(get_console_handler())
    logger.propagate = False
    return logger

