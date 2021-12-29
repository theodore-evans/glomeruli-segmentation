from logging import Logger
from typing import Optional, Tuple

import numpy as np
from app.data_types import Rectangle, Tile, TileRequest
from app.logging_tools import get_logger
from data.dataset import Dataset
from numpy import ndarray
from PIL import Image


class TileLoader(Dataset):
    def __init__(
        self,
        tile_request: TileRequest,
        wsi_region: Rectangle,
        window_size: int = 1024,
        level: int = 0,
        image_stride: Optional[int] = None,
        logger: Logger = get_logger(),
    ) -> None:
        """
        Description:
            A WSI tile retrieving object, extending data.dataset.Dataset, \
        taking an API call for a given WSI and providing an iterator over square tiles \
        of length <window_size> with type numpy.ndarray

        Arguments:
            tile_request: API call to fetch a specified WSI tile in Image format, see app.data_types.TileRequest
            wsi_region: rectangle defining the region of a WSI from which to fetch tiles
            window_size: The side length of the tiles to fetch from the API
            image_stride: Optional pixel offset between subsequent tiles, defaults to window_size if None

        """
        self.tile_request = tile_request
        self.window_size = window_size
        self.level = level
        self.image_stride = image_stride if image_stride is not None else window_size
        self.logger = logger
        self.wsi_region = wsi_region

        # self.cols: int = self.width - self.window_size
        # self.rows: int = self.height - self.window_size

        self.offset_x = list(
            range(
                self.origin[0],
                self.origin[0] + self.width - self.window_size + self.image_stride,
                self.image_stride,
            )
        )
        self.offset_y = list(
            range(
                self.origin[1],
                self.origin[1] + self.height - self.window_size + self.image_stride,
                self.image_stride,
            )
        )

        self.offset_x[-1] = self.shift_offset_into_image(self.offset_x[-1], self.origin[0] + self.width)
        self.offset_y[-1] = self.shift_offset_into_image(self.offset_y[-1], self.origin[1] + self.height)

    @property
    def origin(self) -> Tuple[int, int]:
        """Wsi region origin (upper left in wsi coordinates"""
        return self.wsi_region["upper_left"]

    @property
    def height(self) -> int:
        """Original wsi image height"""
        return self.wsi_region["height"]

    @property
    def width(self) -> int:
        """Original wsi image width"""
        return self.wsi_region["width"]

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

    def shift_offset_into_image(self, x: int, max_x: int) -> int:
        """
        Shift offset to make entire window lie within the image
        """
        if x + self.window_size > max_x:
            x = max_x - self.window_size
        return x

    def __getitem__(self, index: int) -> Tile:
        return self.getitem(index)

    def getitem(self, index: int) -> Tile:
        """
        Retrieve a dictionary containing image data and x and y
        coordinates for the position of the bottom left corner of the tile
        """
        x, y = self.get_offset(index)

        tile_rectangle: Rectangle = {
            "upper_left": [x, y],
            "width": self.window_size,
            "height": self.window_size,
            "level": self.level,
        }

        wsi_tile: Image.Image = self.tile_request(tile_rectangle)
        tile_as_array: np.ndarray = np.asarray_chkfinite(wsi_tile).squeeze()

        # try:
        #     tile_as_array = np.transpose(tile_as_array, (1, 2, 0))
        # except ValueError as e:
        #     self.logger.error(
        #         f"""
        #         Expected image data with 3 channels, instead got {tile_as_array.shape[2]}
        #         """
        #     )
        #     raise e

        tile: Tile = {"image": tile_as_array, "x": x, "y": y}
        return tile

    def __len__(self) -> int:
        return len(self.offset_x) * len(self.offset_y)
