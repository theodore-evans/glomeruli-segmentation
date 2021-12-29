from typing import Callable, List, Tuple

from numpy import ndarray
from PIL.Image import Image
from typing_extensions import TypedDict


class Vector3(TypedDict):
    x: int
    y: int
    z: int


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


class Rectangle(TypedDict):
    upper_left: Tuple[int, int]
    width: int
    height: int
    level: int


TileRequest = Callable[[Rectangle], Image]


class Tile(TypedDict):
    image: ndarray
    x: int
    y: int
