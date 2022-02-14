from sys import maxsize
from typing import Collection, Iterable

import numpy as np

from glomeruli_segmentation.data_classes import Rectangle, Tile


def get_bounds(
    rectangles: Iterable[Rectangle],
) -> Rectangle:
    """
    Finds the bounding rectangle for a collection of rectangles
    """

    x_max = y_max = -maxsize - 1
    x_min = y_min = maxsize

    level = 0

    for rect in rectangles:
        x, y = rect.upper_left
        x_min = min(x, x_min)
        x_max = max(x + rect.width, x_max)
        y_min = min(y, y_min)
        y_max = max(y + rect.height, y_max)
        level = rect.level

    bounds = Rectangle(upper_left=(x_min, y_min), width=x_max - x_min, height=y_max - y_min, level=level)
    return bounds


def combine_masks(
    tiles: Collection[Tile],
) -> Tile:
    """
    Combine a collection of tiles into one big tile
    Overlapping image pixels are averaged elementwise
    """
    bounds = get_bounds((tile.rect for tile in tiles))
    x_min, y_min = bounds.upper_left
    combined = np.zeros((bounds.height, bounds.width))
    normalization = np.zeros_like(combined)

    for tile in tiles:
        x, y = tile.rect.upper_left
        x_start = x - x_min
        x_end = x_start + tile.rect.width
        y_start = y - y_min
        y_end = y_start + tile.rect.height
        combined[y_start:y_end, x_start:x_end] += tile.image
        normalization[y_start:y_end, x_start:x_end] += 1

    if np.min(normalization) == 0:
        raise ValueError("Divide by zero in tile combination")
    combined /= normalization

    return Tile(image=combined, rect=bounds)
