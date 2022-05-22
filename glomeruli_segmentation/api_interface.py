from io import BytesIO
from logging import Logger
from typing import Tuple, Type, Union

import desert
from aiohttp import ClientSession, TCPConnector
from marshmallow import EXCLUDE
from PIL import Image

from glomeruli_segmentation.aiohttp_hooks import get_logging_hooks
from glomeruli_segmentation.data_classes import Rect, Tile, Wsi
from glomeruli_segmentation.logging_tools import get_logger

API_VERSION = "v0"


class LoggingClientSession(ClientSession):
    def __init__(self, logger: dict, get_hooks: Tuple = get_logging_hooks, **kwargs):
        super().__init__(**kwargs)
        self.hooks = {"response": [get_hook(logger) for get_hook in get_hooks]}

    async def _request(self, method, str_or_url, **kwargs):
        r = await super()._request(method, str_or_url, **kwargs)
        for hook in self.hooks["response"]:
            await hook(r)
        return r


# TODO: add methods for getting from /configuration and post/putting to /failure
class ApiInterface:
    def __init__(self, api_url: str, job_id: str, headers: dict, logger: Logger = get_logger()):
        self.logger = logger
        self.api_url = api_url
        self.job_id = job_id
        self.headers = headers
        self.session: LoggingClientSession = None

    async def __aenter__(self):
        self.session = LoggingClientSession(
            connector=TCPConnector(keepalive_timeout=5, ssl=False, limit=10),
            headers=self.headers,
            logger=self.logger,
        )
        await self.session.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            return await self.session.__aexit__(exc_type, exc_val, exc_tb)

    async def check_alive(self) -> dict:
        """
        check if API is alive
        """
        url = f"{self.api_url}/alive"
        r = await self.session.get(url)
        return await r.json()

    async def get_input(self, key: str, input_type: Type) -> Union[Rect, Wsi]:
        """
        fetch an input from API
        """
        url = f"{self.api_url}/{API_VERSION}/{self.job_id}/inputs/{key}"
        r = await self.session.get(url)

        schema = desert.schema(input_type, meta={"unknown": EXCLUDE})
        response = await r.json()
        return schema.load(response)

    async def post_output(self, key: str, data: dict) -> dict:
        """
        post output data by key as defined in EAD
        """
        url = f"{self.api_url}/{API_VERSION}/{self.job_id}/outputs/{key}"
        r = await self.session.post(url, json=data)
        return await r.json()

    async def post_items_to_collection(self, collection: dict, items: list) -> dict:
        """
        add items to an existing output collection

        Parameters:
            items: list of data elements
        """
        url = f"{self.api_url}/{API_VERSION}/{self.job_id}/collections/{collection['id']}/items"
        r = await self.session.post(url, json={"items": items})
        items = await r.json()
        return items["items"]

    async def get_wsi_tile(self, slide: Wsi, rect: Rect) -> Tile:
        """
        get a WSI tile on level 0

        Parameters:
            wsi_slide: Wsi object with WSI id (and meta data)
            rectangle: Rectangle describing tile position
        """
        x, y = rect.upper_left

        url = f"{self.api_url}/{API_VERSION}/{self.job_id}/regions/{slide.id}/level/{rect.level}/start/{x}/{y}/size/{rect.width}/{rect.height}"

        r = await self.session.get(url)
        content = await r.read()
        return Tile(image=Image.open(BytesIO(content)), rect=rect)

    async def put_finalize(self) -> dict:
        """
        finalize job, such that no more data can be added and to inform EMPAIA infrastructure about job state
        """
        url = f"{self.api_url}/{API_VERSION}/{self.job_id}/finalize"

        r = await self.session.put(url)
        return await r.json()

    async def put_failure(self, message: str) -> dict:

        url = f"{self.api_url}/{API_VERSION}/{self.job_id}/failure"
        r = await self.session.put(url, json={"user_message": message.replace('"', "'")})
        return await r.json()
