from typing import Tuple
import PIL.Image as Image
import tifffile
import numpy as np
from app.data_types import Rectangle, WSI, Vector3, Level
import os
import json

kidney_wsi_id = "37bd11b8-3995-4377-bf57-e718e797d515"
rectangle_id = "37bd11b8-3995-4377-bf57-e718e797d516"
APP_API = os.environ["EMPAIA_APP_API"]
JOB_ID = os.environ["EMPAIA_JOB_ID"]
TOKEN = os.environ["EMPAIA_TOKEN"]


class MockAPI:
    def __init__(self, sample_image_file: str):
        self.image_data = tifffile.imread(sample_image_file).squeeze()

    def mock_tile_request(self, rectangle: Rectangle):
        x, y = rectangle['upper_left']
        width = rectangle['width']
        height = rectangle['height']
        return self.image_data[:, x:x+width, y:y+height]

    def get_input(self, key: str):
        if key == "kidney_wsi":
            extent = Vector3(x=0, y=0, z=0)
            return WSI(id=kidney_wsi_id, extent=extent, num_levels=1, pixel_size_nm=extent, tile_extent=extent, levels=[Level(extent=extent, downsample_factor=1, generated=True)])
        elif key == "my_rectangle":
            return Rectangle(id=rectangle_id, upper_left=[15000, 7000], width=4096, height=4096)

    def post_output(self, key: str, data: dict) -> dict:
        """
        post output data by key as defined in EAD
        """
        data["creator_id"] = "dummy_creator_id"
        data["creator_type"] = "job"
        with open(f'/app/outputs/{key}.json', 'w+') as outfile:
            json.dump(data, outfile, indent=2)

    def get_wsi_tile(self, my_wsi: WSI, my_rectangle: Rectangle):
        if my_wsi["id"] == kidney_wsi_id:
            return self.mock_tile_request(my_rectangle)

    def put_finalize(self):
        """
        finalize job, such that no more data can be added and to inform EMPAIA infrastructure about job state
        """
        print("Job Finished")
