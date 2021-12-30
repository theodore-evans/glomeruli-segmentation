import json
import os

import numpy as np
from app.data_types import WSI, Level, Rectangle, Vector3

KIDNEY_WSI_ID = "37bd11b8-3995-4377-bf57-e718e797d515"
RECT_ID = "37bd11b8-3995-4377-bf57-e718e797d516"

OUTPUT_DIRECTORY = "../outputs"


class MockAPI:
    def __init__(self, image_data: np.ndarray):
        self.image_data = image_data

    def get_input(self, key: str):
        if key == "slide":
            extent = Vector3(x=self.image_data.shape[0], y=self.image_data.shape[1], z=1)
            pixel_size = Vector3(x=500, y=500, z=1)
            return WSI(
                id=KIDNEY_WSI_ID,
                extent=extent,
                num_levels=1,
                pixel_size_nm=pixel_size,
                tile_extent=extent,
                levels=[Level(extent=extent, downsample_factor=1, generated=False)],
            )
        elif key == "region_of_interest":
            return Rectangle(id=RECT_ID, upper_left=(1000, 1000), width=3000, height=1500)

    def post_output(self, key: str, data: dict) -> dict:
        """
        post output data by key as defined in EAD
        """
        data["creator_id"] = "dummy_creator_id"
        data["creator_type"] = "job"

        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

        with open(f"{OUTPUT_DIRECTORY}/{key}.json", "w+") as outfile:
            json.dump(data, outfile, indent=2)

        print(data)
        return data

    def get_wsi_tile(self, slide: WSI, rectangle: Rectangle):
        if slide["id"] == KIDNEY_WSI_ID:
            x, y = rectangle["upper_left"]
            width = rectangle["width"]
            height = rectangle["height"]
            return self.image_data[x : x + width, y : y + height, :]
        else:
            raise ValueError(f"no such slide {slide}")

    def put_finalize(self):
        """
        finalize job, such that no more data can be added and to inform EMPAIA infrastructure about job state
        """
        print("Job Finished")
