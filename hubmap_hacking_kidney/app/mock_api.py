import json
import os
from typing import Tuple

import numpy as np
import PIL.Image as Image
import tifffile
from app.data_types import WSI, Level, Rectangle, Vector3

KIDNEY_WSI_ID = "37bd11b8-3995-4377-bf57-e718e797d515"
RECT_ID = "37bd11b8-3995-4377-bf57-e718e797d516"

SAMPLE_IMAGE_FILE = "/empaia/data/real_kidney.tif"
OUTPUT_DIRECTORY = "./outputs"


class MockAPI:
    def __init__(self):
        self.image_data = tifffile.imread(SAMPLE_IMAGE_FILE).squeeze()

    def mock_tile_request(self, rectangle: Rectangle):
        x, y = rectangle["upper_left"]
        width = rectangle["width"]
        height = rectangle["height"]
        return self.image_data[x : x + width, y : y + height, :]

    def get_input(self, key: str):
        if key == "kidney_wsi":
            extent = Vector3(x=22240, y=30440, z=1)
            pixel_size = Vector3(x=500, y=500, z=1)
            return WSI(
                id=KIDNEY_WSI_ID,
                extent=extent,
                num_levels=1,
                pixel_size_nm=pixel_size,
                tile_extent=extent,
                levels=[Level(extent=extent, downsample_factor=1, generated=False)],
            )
        elif key == "my_rectangle":
            return Rectangle(id=RECT_ID, upper_left=[15000, 7000], width=2048, height=2048)

    def post_output(self, key: str, data: dict) -> dict:
        """
        post output data by key as defined in EAD
        """
        data["creator_id"] = "dummy_creator_id"
        data["creator_type"] = "job"

        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

        with open(f"{OUTPUT_DIRECTORY}/{key}.json", "w+") as outfile:
            json.dump(data, outfile, indent=2)

        return data

    def get_wsi_tile(self, my_wsi: WSI, my_rectangle: Rectangle):
        if my_wsi["id"] == KIDNEY_WSI_ID:
            return self.mock_tile_request(my_rectangle)

    def put_finalize(self):
        """
        finalize job, such that no more data can be added and to inform EMPAIA infrastructure about job state
        """
        print("Job Finished")
