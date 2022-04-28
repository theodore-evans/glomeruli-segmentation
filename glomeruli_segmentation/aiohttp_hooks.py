import json
import logging
from typing import Callable

import aiohttp
from aiohttp import ClientResponse, ContentTypeError

Hook = Callable[[ClientResponse], None]


def _pretty_json(response_json):
    return json.dumps(response_json, indent=4, sort_keys=True)


MBFACTOR = float(1 << 20)


def _bytes_to_mb(size_in_bytes: int) -> float:
    return round(size_in_bytes / MBFACTOR, 3)


def get_raise_for_status_hook(logger: logging.Logger) -> Hook:
    """
    Requests hook to raise an exception if the response status is not 2xx

    :param logger: Logger to use for logging.
    :return: Hook
    """

    async def hook(r: ClientResponse, *args, **kwargs):
        try:
            # FIXME?: this will close the session if an exception is raised
            # Do not know whether this is desired behavior? Certainly easier for debugging
            # Maybe implement a call to /failure endpoint?
            r.raise_for_status()
        except aiohttp.ClientResponseError as e:
            logger.error(f'Error during "{r.method} {r.url}": {e.status} {e.message}')

    return hook


def get_log_response_hook(logger: logging.Logger) -> Hook:
    """
    Requests hook to log the response details, depending on log level

    :param logger: Logger to use for logging.
    :return: Hook
    """

    async def hook(r: ClientResponse, *args, **kwargs):
        logger.info(f'"{r.method} {r.url}" {r.status} {r.reason}')
        if logger.level <= logging.DEBUG:
            logger.debug(f"headers={_pretty_json(dict(r.headers))}")
            try:
                response_json = await r.json()
                logger.debug(f"response={_pretty_json(response_json)}")
            except TypeError:
                response_text = await r.text()
                logger.debug(f"response={response_text}")
            except ContentTypeError:
                logger.debug(f"response={r.content_type} size={_bytes_to_mb(r.content.total_bytes)}MB")

    return hook


get_logging_hooks = (get_raise_for_status_hook, get_log_response_hook)
