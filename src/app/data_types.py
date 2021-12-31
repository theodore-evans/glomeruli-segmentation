from typing import Callable, List, Tuple

from numpy import ndarray
from PIL.Image import Image
from typing_extensions import TypedDict


class Vector3(TypedDict):
    x: int
    y: int
    z: int


class Vector2:
    x: int
    y: int

    def __init__(self, value: Tuple[int, int]):
        x, y = value


class Level(TypedDict):
    extent: Vector3
    downsample_factor: int
    generated: bool


class WSI(TypedDict):
    id: str
    extent: Vector3
    num_levels: int
    pixel_size_nm: Vector3
    tile_extent: Vector3
    levels: List[Level]


# TODO: refactor this (these two?) into dataclass(es) and then refactor tests and methods
class Rectangle(TypedDict):
    upper_left: Tuple[int, int]
    width: int
    height: int
    level: int


class Tile(TypedDict):
    image: ndarray
    rect: Rectangle


TileGetter = Callable[[Rectangle], Tile]
