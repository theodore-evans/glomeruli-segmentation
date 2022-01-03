from dataclasses import dataclass
from typing import List, Tuple

from numpy import array_equal, ndarray


@dataclass
class Vector3:
    x: int
    y: int
    z: int


@dataclass
class Vector2:
    x: int
    y: int

    def __getitem__(self, key):
        return (self.x, self.y)[key]

    def __iter__(self):
        return (value for value in (self.x, self.y))


@dataclass
class Level:
    extent: Vector3
    downsample_factor: int
    generated: bool


@dataclass
class Wsi:
    id: str
    extent: Vector3
    num_levels: int
    pixel_size_nm: Vector3
    tile_extent: Vector3
    levels: List[Level]
    # add additional non-required fields


@dataclass
class Rectangle:
    upper_left: Vector2
    width: int
    height: int
    level: int = 0
    id: str = ""
    
    def __post_init__(self):
        self.upper_left = Vector2(*self.upper_left)

    @property
    def shape(self) -> Tuple:
        return (self.width, self.height)


@dataclass
class Tile:
    image: ndarray
    rect: Rectangle

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return array_equal(self.image, other.image) and self.rect == other.rect
        else:
            return False
