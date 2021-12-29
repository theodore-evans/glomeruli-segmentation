from io import BytesIO
from logging import Logger

import requests
from app.logging_tools import get_logger
from PIL import Image
from request_hooks import check_for_errors_hook, response_logging_hook


class ApiInterface:
    def __init__(self, api_url: str, job_id: str, headers: str, logger: Logger = get_logger()):

        self.logger = logger
        self.api_url = api_url
        self.job_id = job_id
        self.headers = headers

        self.logger.info(f"{self.api_url=} {self.job_id=}")

        self.session = requests.Session()
        request_hooks = [check_for_errors_hook, response_logging_hook]
        self.session.hooks["response"] = [hook(self.logger) for hook in request_hooks]

    def get_input(self, key: str) -> dict:
        """
        get input data by key as defined in EAD
        """
        url = f"{self.api_url}/v0/{self.job_id}/inputs/{key}"
        r = self.session.get(url, headers=self.headers)

        return r.json()

    def post_output(self, key: str, data: dict) -> dict:
        """
        post output data by key as defined in EAD
        """
        url = f"{self.api_url}/v0/{self.job_id}/outputs/{key}"
        r = self.session.post(url, json=data, headers=self.headers)

        return r.json()

    def get_wsi_tile(self, wsi_slide: dict, rectangle: dict) -> Image.Image:
        """
        get a WSI tile on level 0

        Parameters:
            wsi_slide: contains WSI id (and meta data)
            rectangle: tile position on level 0
        """
        x, y = rectangle["upper_left"]
        width = rectangle["width"]
        height = rectangle["height"]

        wsi_id = wsi_slide["id"]
        level = 0

        tile_url = f"{self.api_url}/v0/{self.job_id}/regions/{wsi_id}/level/{level}/start/{x}/{y}/size/{width}/{height}"

        r = self.session.get(tile_url, headers=self.headers)

        return Image.open(BytesIO(r.content))

    def put_finalize(self):
        """
        finalize job, such that no more data can be added and to inform EMPAIA infrastructure about job state
        """
        url = f"{self.api_url}/v0/{self.job_id}/finalize"
        r = self.session.put(url, headers=self.headers)
