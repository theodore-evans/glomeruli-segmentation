from typing import Dict, List, Optional, Tuple, Union
from typing_extensions import TypedDict

import numpy as np
from numpy import ndarray
import cv2
from PIL import Image
import torch
from torch._C import int64

from data.dataset import Dataset

class Tile(TypedDict):
    image: ndarray
    x: int
    y: int
    
class WSITileDataset(Dataset):
    def __init__(self,
                 wsi_tile: Image.Image,
                 window_size: int = 1024,
                 image_stride: Optional[int] = None,
                 scale: Optional[float] = None
                 ) -> None:
        """
        Instantiate a WSI tile parsing object, extending data.dataset.Dataset,
        taking a PIL.Image.Image object and retrieving square tiles
        of length window_size as ndarrays
            
        Arguments:
            wsi_tile: WSI image or image region in PIL.Image.Image format
            window_size: The side length of the tiles extracted from wsi_image
            image_stride: Optional pixel offset between subsequent tiles, defaults to window_size if None
            scale: Optional scaling of image before processing
            
        """
        self.scale = scale
        self.window_size = window_size
        self.image_stride = window_size if image_stride is None else image_stride
        self.image = np.asarray_chkfinite(wsi_tile).squeeze()
        
        if self.image.shape[0] == 3:
            self.image = np.transpose(self.image, (1, 2, 0))
        assert self.image.shape[2] == 3 #TODO: error handling here
        
        self.original_width = self.image.shape[1]
        self.original_height = self.image.shape[0]
        
        self.cols = self.width - self.window_size
        self.rows = self.height - self.window_size
        
        self.offset_x = list(range(0, self.width - self.window_size + self.image_stride, self.image_stride))
        self.offset_y = list(range(0, self.height - self.window_size + self.image_stride, self.image_stride))
        
        self.offset_x[-1] = self.shift_offset_into_image(self.offset_x[-1], self.width)
        self.offset_y[-1] = self.shift_offset_into_image(self.offset_y[-1], self.height)
        
        if scale:
            self.image = cv2.resize(self.image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]

    def read(self, x: int, y: int, width: int, height: int) -> ndarray:
        """Get a slice of image data from y:y + height, x:x + width"""
        return self.image[y:y + height, x:x + width]
    
    def shift_offset_into_image(self, x: int, max_x: int) -> int:
        """
        Shift offset to make entire window lie within the image
        """
        if x + self.window_size > max_x:
            x = max_x - self.window_size
        return x
    
    def getitem(self, index: int) -> Tile:
        """
        Retrieve a dictionary containing image data and x and y
        coordinates for the position of the bottom left corner of the tile
        """
        x, y = self.get_offset(index)
        image = self.read(x, y, self.window_size, self.window_size)
        tile: Tile = {"image":image, "x":x, "y":y}
        return tile
    
    def __len__(self) -> int:
        return len(self.offset_x) * len(self.offset_y)
    
    def __getitem__(self, index: int) -> Tile:
        return self.getitem(index)
    
    def get_offset(self, index: int) -> Tuple[int, int]:
        """
        Get pixel coordinates for a given tile index
        """
        width = len(self.offset_x)
        y_i = index // width
        x_i = index % width
        x = self.offset_x[x_i]
        y = self.offset_y[y_i]
        return x, y
    
def combine_tiles(tiles: List[Tile],
                  original_height: int,
                  original_width: int,
                  ignore_border=0):
        """
        Stitch tiles to an array of size original_height, original_width
        """
        combined_image = torch.zeros((original_height, original_width), dtype=torch.float32) #type: ignore
        n = torch.zeros((original_height, original_width), dtype=torch.uint8) #type: ignore
        
        for tile in tiles:
            image = tile["image"]
            x = tile["x"]
            y = tile["y"]

            w = image.shape[1]
            h = image.shape[0]
            xx = x + w
            yy = y + h
            l = 0
            t = 0

            if ignore_border > 0:
                if x > 0:
                    x += ignore_border
                    l = ignore_border
                if xx < original_width:
                    xx -= ignore_border
                if y > 0:
                    y += ignore_border
                    t = ignore_border
                if yy < original_height:
                    yy -= ignore_border
                w = xx - x + l
                h = yy - y + t
                
            combined_image[y:yy, x:xx] += image[t:h, l:w]
            n[y:yy, x:xx] += 1
        
        combined_image /= n
        return combined_image