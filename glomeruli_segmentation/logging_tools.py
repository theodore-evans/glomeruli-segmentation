import logging
import sys

DEFAULT_FORMATTER = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")


def get_log_level(verbosity: int = 0, default_log_level: int = logging.WARN, min_log_level: int = logging.DEBUG) -> int:
    return max(default_log_level - 10 * verbosity, min_log_level)


def get_console_handler(formatter: logging.Formatter = DEFAULT_FORMATTER) -> logging.Handler:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    return console_handler


def get_logger(
    logger_name: str = __name__,
    log_level: int = get_log_level(0),
    propagate: bool = False,
    handler: logging.Handler = get_console_handler(),
) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.addHandler(handler)
    logger.propagate = propagate
    return logger
