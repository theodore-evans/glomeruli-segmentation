from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

import numpy as np
from numpy import array_equal, ndarray
from PIL.Image import Image
from torch import Tensor


class BlendMode(Enum):
    OVERWRITE = "overwrite"
    MAX = "max"
    MEAN = "mean"


def average_non_nan(a, b):
    """
    Overwrite all nan elements of a with the corresponding elements of b.
    Average all non-nan elements of a with the corresponding elements of b.
    """
    a[np.isnan(a)] = b[np.isnan(a)]
    a[~np.isnan(a)] = np.add(a[~np.isnan(a)], b[~np.isnan(a)]) / 2
    return a


class Vector2(List):
    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]


@dataclass
class Vector3:
    x: int
    y: int
    z: Optional[int] = None

    def __iter__(self):
        return iter(c for c in (self.x, self.y, self.z) if c is not None)

    def __getitem__(self, index):
        return list(self)[index]


@dataclass
class Level:
    extent: Vector3
    downsample_factor: int


@dataclass
class Wsi:
    extent: Vector3
    num_levels: int
    pixel_size_nm: Vector3
    tile_extent: Vector3
    levels: List[Level]
    id: str = ""


@dataclass
class Rectangle:
    upper_left: List[int]
    width: int
    height: int
    level: int = 0
    id: str = ""

    def __post_init__(self):
        self.upper_left = Vector2((self.upper_left))
        return (self.width, self.height)

    @property
    def shape(self):
        """
        Returns the rectangle shape as a tuple of (height, width)
        """
        return (self.height, self.width)


@dataclass
class Tile:
    image: Union[Image, ndarray, Tensor]
    rect: Rectangle
    id: str = ""

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return array_equal(self.image, other.image) and self.rect == other.rect
        else:
            return False


@dataclass
class Mask(Tile):
    """
    A Mask is a tile with a single channel mask as an image.
    """

    @classmethod
    def empty(cls, rect: Rectangle) -> "Mask":
        """
        Creates an empty mask with the given rectangle.

        :param rect: The rectangle to create the mask for
        :param id: The id of the mask, defaults to ""
        """
        image = np.full(rect.shape, np.nan)
        return Mask(image=image, rect=rect, id=id)

    def insert_patch(self, new_patch: "Tile", blend_mode: BlendMode = BlendMode.OVERWRITE) -> "Tile":
        """
        Inserts a new patch into the current mask.

        :param new_patch: The patch to insert
        :param blend_mode: How to blend overlapping data from the new patch with the current mask, options: OVERWRITE, MAX, MEAN
        """
        x_min, y_min = self.rect.upper_left
        x, y = new_patch.rect.upper_left
        x_start = x - x_min
        x_end = x_start + new_patch.rect.width
        y_start = y - y_min
        y_end = y_start + new_patch.rect.height

        if blend_mode == BlendMode.OVERWRITE:
            blend = lambda _, new: new
        elif blend_mode == BlendMode.MAX:
            blend = np.fmax
        elif blend_mode == BlendMode.MEAN:
            blend = average_non_nan
        else:
            raise NotImplementedError(f"Blend mode '{blend_mode.name}' not implemented")

        current_patch_data = self.image[y_start:y_end, x_start:x_end].squeeze()
        new_patch_data = new_patch.image.squeeze()
        assert (
            new_patch_data.shape == current_patch_data.shape
        ), f"Patch size mismatch: {new_patch_data.shape} != {current_patch_data.shape}"

        self.image[y_start:y_end, x_start:x_end] = blend(current_patch_data, new_patch_data)
