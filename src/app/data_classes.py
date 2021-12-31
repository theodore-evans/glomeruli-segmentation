from typing import List, Optional, Tuple, Union

from numpy import ndarray
from dataclasses import dataclass


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
        return (self.x,self.y)[key]
    def __iter__(self):
        return (value for value in (self.x, self.y))

# TODO: refactor below into dataclasses and then refactor tests and methods
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
    level: int
    # add additional non-required fields

    def __post_init__(self):
        self.upper_left = Vector2(*self.upper_left)


@dataclass
class Tile:
    image: ndarray
    rect: Rectangle
