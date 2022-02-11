from io import BytesIO
from logging import Logger
from typing import Type, Union

import desert
import requests
from marshmallow import EXCLUDE, INCLUDE, ValidationError
from PIL import Image
from requests import Response

from glomeruli_segmentation.data_classes import Rectangle, Tile, Wsi
from glomeruli_segmentation.logging_tools import get_logger
from glomeruli_segmentation.request_hooks import check_for_errors_hook, response_logging_hook


class ApiInterface:
    def __init__(self, api_url: str, job_id: str, headers: dict, logger: Logger = get_logger()):

        self.logger = logger
        self.api_url = api_url
        self.job_id = job_id
        self.headers = headers

        self.session = requests.Session()
        request_hooks = [check_for_errors_hook, response_logging_hook]
        self.session.hooks["response"] = [hook(self.logger) for hook in request_hooks]

    # FIXME: marshmallow.exceptions.ValidationError: {'upper_left': {'_schema': ['Invalid input type.']}}
    def get_input(self, key: str, data_class: Type) -> Union[Rectangle, Wsi]:

        url = f"{self.api_url}/v0/{self.job_id}/inputs/{key}"
        resp = self.session.get(url, headers=self.headers)
        response_json = resp.json()
        try:
            schema = desert.schema(data_class, meta={"unknown": EXCLUDE})
            return schema.load(response_json)
        except:
            self.logger.error(response_json)
            raise

    def post_output(self, key: str, data: dict) -> Response:
        """
        post output data by key as defined in EAD
        """
        url = f"{self.api_url}/v0/{self.job_id}/outputs/{key}"
        resp = self.session.post(url, json=data, headers=self.headers)

        return resp

    def get_wsi_tile(self, rect: Rectangle, slide: Wsi) -> Tile:
        """
        get a WSI tile on level 0

        Parameters:
            wsi_slide: Wsi object with WSI id (and meta data)
            rectangle: Rectangle describing tile position
        """
        x, y = rect.upper_left

        url = f"{self.api_url}/v0/{self.job_id}/regions/{slide.id}/level/{rect.level}/start/{x}/{y}/size/{rect.width}/{rect.height}"

        resp = self.session.get(url, headers=self.headers)
        return Tile(image=Image.open(BytesIO(resp.content)), rect=rect)

    def put_finalize(self) -> Response:
        """
        finalize job, such that no more data can be added and to inform EMPAIA infrastructure about job state
        """
        url = f"{self.api_url}/v0/{self.job_id}/finalize"
        resp = self.session.put(url, headers=self.headers)

        return resp
