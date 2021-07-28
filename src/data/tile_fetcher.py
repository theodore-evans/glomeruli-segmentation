"""CEM Tile Fetcher Module"""
from typing import Optional, Tuple
from nptyping import NDArray, Int

import numpy as np
from cem.data.data_types import Rectangle, Tile, TileRequest
from cem.data.dataset import Dataset
from PIL import Image


class TileFetcher(Dataset):
    def __init__(
        self,
        tile_request: TileRequest,
        wsi_region: Rectangle,
        window_size: int = 1024,
        level: int = 0,
        image_stride: Optional[int] = None,
    ) -> None:
        """ 
        Description:
            A WSI tile retrieving object, extending data.dataset.Dataset, \
            taking an API call for a given WSI. \
            Moreover, holds the most recent delta for the CEM, providing an iterator over square \
            tiles of length <window_size> with type numpy.ndarray. 
            Thus, facilitating data conforming with the model input shape.

        Args:
            tile_request (TileRequest): API call to fetch a specified WSI tile in Image format, see app.data_types.TileRequest
            wsi_region (Rectangle): rectangle defining the region of a WSI from which to fetch tiles
            window_size (int, optional): The side length of the tiles to fetch from the region. Defaults to 1024.
            level (int, optional): Level of the region to be fetched. Defaults to 0.
            image_stride (Optional[int], optional): Optional pixel offset between subsequent tiles. Defaults to None.
        """
        self.tile_request = tile_request
        self.window_size = window_size
        self.level = level
        self.image_stride = image_stride if image_stride is not None else window_size

        self.upper_left = (0, 0)  # region upper left
        self.original_upper_left = wsi_region[
            "upper_left"
        ]  # upper left coordinates in orig image
        self.original_size = (wsi_region["width"], wsi_region["height"])

        self.cols: int = self.width - self.window_size
        self.rows: int = self.height - self.window_size

        self.offset_x = list(
            range(
                0,
                self.width - self.window_size + self.image_stride,
                self.image_stride,
            )
        )
        self.offset_y = list(
            range(
                0,
                self.height - self.window_size + self.image_stride,
                self.image_stride,
            )
        )

        self.offset_x[-1] = self.shift_offset_into_image(self.offset_x[-1], self.width)
        self.offset_y[-1] = self.shift_offset_into_image(self.offset_y[-1], self.height)

        self.fetch_region()

    @property
    def height(self) -> int:
        """Original wsi image height"""
        return self.original_size[1]

    @property
    def width(self) -> int:
        """Original wsi image width"""
        return self.original_size[0]

    def fetch_region(self):
        """Fetch a region of the original image data from the
        EMPAIA API using the tile_request call provided in the constructor
        """
        tile_rectangle: Rectangle = {
            "upper_left": self.original_upper_left,
            "width": self.original_size[0],
            "height": self.original_size[1],
            "level": self.level,
        }

        region: Image.Image = self.tile_request(tile_rectangle)
        region_as_array = np.asarray_chkfinite(region).squeeze()

        if region_as_array.shape[0] == 3:
            region_as_array = np.transpose(region_as_array, (1, 2, 0))
        if region_as_array.shape[2] != 3:
            raise ValueError(
                f"""
                Expected image data with 3 channels, instead got {region_as_array.shape[2]}
                """
            )

        self._orig_region: NDArray[Int] = region_as_array
        self.region = self._orig_region

    @property
    def region(self) -> NDArray[Int]:
        """Current region data, could be the original region or recent delta"""
        return self._region

    @region.setter
    def region(self, new_region: NDArray[Int]) -> None:
        """Sets a new region to be splitted into tiles

        Args:
            new_region (NDArray[Int]): [description]

        Raises:
            ValueError: [description]
        """
        if new_region.shape != self._orig_region.shape:
            raise ValueError(
                f"New region data ({new_region.shape}) has to be of same shape "
                f"as the original region ({self._orig_region.shape})!"
            )
        self._region = new_region

    def reset_region(self) -> None:
        """Resets the currently saved region to the original one"""
        self._region = self._orig_region

    def shift_offset_into_image(self, x: int, max_x: int) -> int:
        """Shift offset to make entire window lie within the image

        Args:
            x (int): x/y-coordinate
            max_x (int): max x/y-coordinate

        Returns:
            int: new offset coordinate within the image
        """
        if x + self.window_size > max_x:
            x = max_x - self.window_size
        return x

    def getitem(self, index: int) -> Tile:
        """Retrieve a dictionary containing image data and x and y \
            coordinates for the position of the bottom left corner of the tile \
            relative to the region

        Args:
            index (int): index of the tile

        Returns:
            Tile: image, relative x and y coordinates
        """
        x, y = self.get_offset(index)
        image = self._region[x : x + self.window_size, y : y + self.window_size, :]
        tile: Tile = {"image": image, "x": x, "y": y}
        return tile

    def __len__(self) -> int:
        return len(self.offset_x) * len(self.offset_y)

    def __getitem__(self, index: int) -> Tile:
        return self.getitem(index)

    def get_offset(self, index: int) -> Tuple[int, int]:
        """Get pixel coordinates for a given tile index

        Args:
            index (int): index of the tile

        Returns:
            Tuple[int, int]: relative x and y coordinates
        """
        width = len(self.offset_x)
        y_i = index // width
        x_i = index % width
        x = self.offset_x[x_i]
        y = self.offset_y[y_i]
        return x, y
