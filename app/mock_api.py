from typing import Tuple
import PIL.Image as Image
import tifffile
import numpy as np
from app.data_types import Rectangle, WSI

class MockAPI:
    def __init__(self, sample_image_file: str):   
        self.image_data = tifffile.imread(sample_image_file).squeeze()

    def mock_tile_request(self, rectangle: Rectangle):
        x, y = rectangle['upper_left']
        width = rectangle['width']
        height = rectangle['height']
        return self.image_data[:, x:x+width, y:y+height]