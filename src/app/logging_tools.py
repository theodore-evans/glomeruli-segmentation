import logging
import sys

DEFAULT_FORMATTER = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")


def get_log_level(verbosity: int = 0, default_log_level=logging.WARN, min_log_level=logging.DEBUG):
    return max(default_log_level - 10 * verbosity, min_log_level)


def get_console_handler(formatter: logging.Formatter = DEFAULT_FORMATTER):
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    return console_handler


def get_logger(
    logger_name=__name__, log_level=get_log_level(0), propagate=False, handler=get_console_handler()
):
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.addHandler(handler)
    logger.propagate = propagate
    return logger
