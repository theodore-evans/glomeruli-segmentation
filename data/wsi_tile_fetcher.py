from typing import Optional, Tuple

import numpy as np
from numpy import ndarray
from PIL import Image

from data.dataset import Dataset
from app.data_types import Rectangle, Tile, TileRequest
    
class WSITileFetcher(Dataset):
    def __init__(self,
                tile_request: TileRequest,
                original_size: Tuple[int, int],
                window_size: int = 1024,
                level: int = 0,
                image_stride: Optional[int] = None,
                ) -> None:
        """
        Description:
            A WSI tile retrieving object, extending data.dataset.Dataset, \
        taking an API call for a given WSI and providing an iterator over square tiles \
        of length <window_size> with type numpy.ndarray
            
        Arguments:
            tile_request: API call to fetch a specified WSI tile in Image format, see app.data_types.TileRequest
            original_size: Size in pixels (height: int, width: int) of original WSI image
            window_size: The side length of the tiles to fetch from the API
            image_stride: Optional pixel offset between subsequent tiles, defaults to window_size if None
                
        """
        self.tile_request = tile_request
        self.window_size = window_size
        self.level = level
        self.image_stride = image_stride if image_stride is not None else window_size
        
        self.original_size = original_size
        
        self.cols: int = self.width - self.window_size
        self.rows: int = self.height - self.window_size
        
        self.offset_x = list(range(0, self.width - self.window_size + self.image_stride, self.image_stride))
        self.offset_y = list(range(0, self.height - self.window_size + self.image_stride, self.image_stride))
        
        self.offset_x[-1] = self.shift_offset_into_image(self.offset_x[-1], self.width)
        self.offset_y[-1] = self.shift_offset_into_image(self.offset_y[-1], self.height)

    @property
    def height(self) -> int:
        """Original wsi image height"""
        return self.original_size[0]

    @property
    def width(self) -> int:
        """Original wsi image width"""
        return self.original_size[1]

    def fetch_tile(self, x: int, y: int, width: int, height: int) -> ndarray:
        """
        Fetch a tile of image data from y:y + height, x:x + width from the
        EMPAIA API using the tile_request call provided in the constructor
        """
        tile_rectangle: Rectangle = {
            "upper_left": (x, y+height),
            "width": width,
            "height": height,
            "level": self.level}
        
        wsi_tile: Image.Image = self.tile_request(tile_rectangle)
        tile_as_array = np.asarray_chkfinite(wsi_tile).squeeze()
        
        if tile_as_array.shape[0] == 3:
            tile_as_array = np.transpose(tile_as_array, (1, 2, 0))
        if tile_as_array.shape[2] != 3:
            raise ValueError(
                f'''
                Expected image data with 3 channels, instead got {tile_as_array.shape[2]}
                ''')
        
        return tile_as_array
    
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
        image = self.fetch_tile(x, y, self.window_size, self.window_size)
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