import json
import logging
from typing import Callable
from urllib.parse import urlparse

import requests


def check_for_errors_hook(logger: logging.Logger) -> Callable:
    """
    Requests hook to check for errors
    """

    def hook(r: requests.Response, *args, **kwargs):
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error(f"Error during {r.request.method} to {r.url}:")
            error_text = e.response.text
            try:
                parsed_error = json.loads(error_text)
                logger.error(json.dumps(parsed_error, indent=4, sort_keys=True))
            except TypeError:
                logger.error(error_text)
            finally:
                raise e

    return hook


def response_logging_hook(logger: logging.Logger) -> Callable:
    """
    Requests hook to log request details
    """

    def hook(r: requests.Response, *args, **kwargs):
        if logger.level < logging.WARN:
            url = urlparse(r.url)
            logger.info(
                f'{url.scheme}://{url.netloc} "{r.request.method} {url.path}" {r.status_code} {r.reason}'
            )
        if logger.level < logging.INFO:
            logger.debug(f"{r.headers=}")
            try:
                parsed_response = json.loads(r.text)
                logger.debug(f"r.text={json.dumps(parsed_response, indent=4, sort_keys=True)}")
            except TypeError:
                pass

    return hook
