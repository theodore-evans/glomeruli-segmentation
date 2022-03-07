from dataclasses import dataclass
from typing import List, Optional

from numpy import array_equal, ndarray


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
    id: str
    extent: Vector3
    num_levels: int
    pixel_size_nm: Vector3
    tile_extent: Vector3
    levels: List[Level]


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
    image: ndarray
    rect: Rectangle

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return array_equal(self.image, other.image) and self.rect == other.rect
        else:
            return False
