from sys import maxsize
from typing import Collection, Iterable

import padl

from glomeruli_segmentation.data_classes import BlendMode, Mask, Rect, Tile


def get_bounds(
    rectangles: Iterable[Rect],
) -> Rect:
    """
    Finds the bounding rectangle for a collection of rectangles
    """
    x_max = y_max = -maxsize - 1
    x_min = y_min = maxsize

    for rect in rectangles:
        x, y = rect.upper_left
        x_min = min(x, x_min)
        x_max = max(x + rect.width, x_max)
        y_min = min(y, y_min)
        y_max = max(y + rect.height, y_max)
        level = rect.level

    return Rect(upper_left=[x_min, y_min], width=x_max - x_min, height=y_max - y_min, level=level)


def combine_masks(masks: Collection[Tile], blend_mode: BlendMode = BlendMode.OVERWRITE) -> Tile:
    """
    Combine a collection of tiles into one big tile
    Overlapping image pixels are averaged elementwise
    """
    bounds = get_bounds((tile.rect for tile in masks))
    combined = Mask.empty_zarr(bounds)

    for mask in masks:
        combined.insert_patch(mask, blend_mode)

    return combined


def coords_to_int(polygons):
    for polygon in polygons:
        yield [(int(x), int(y)) for x, y in polygon.exterior.coords]


@padl.transform
class Threshold:
    def __init__(self, t: float):
        self.t = t

    def __call__(self, image):
        image[image < self.t] = 0
        return image


def batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]
