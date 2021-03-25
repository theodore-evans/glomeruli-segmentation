from typing import Tuple
import PIL.Image as Image
import tifffile
import numpy as np
from app.data_types import Rectangle

class MockAPI:
    def __init__(self, sample_image_file: str, upper_left: Tuple[int,int], channel_axis = True):
        self.image_data = tifffile.imread(sample_image_file).squeeze()
        self.channel_axis = channel_axis
        self.upper_left = upper_left
        if not channel_axis:
            self.image_data = self.image_data[0]
            
    def mock_tile_request(self, rectangle: Rectangle):
        x, y = rectangle['upper_left']
        x += self.upper_left[0]
        y += self.upper_left[1]
        width = rectangle['width']
        height = rectangle['height']
        return self.image_data[:, x:x+width, y:y+height] if self.channel_axis else self.image_data[x:x+width, y:y+height]
    