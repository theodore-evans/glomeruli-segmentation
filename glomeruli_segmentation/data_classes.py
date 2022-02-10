from collections import namedtuple
from dataclasses import dataclass
from typing import List, NamedTuple, Tuple

from numpy import array_equal, ndarray


class Vector3(NamedTuple):
    x: int
    y: int
    z: int


class Vector2(NamedTuple):
    x: int
    y: int


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
