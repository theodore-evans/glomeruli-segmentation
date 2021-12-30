from sys import maxsize
from typing import Collection

import numpy as np
from app.data_types import Rectangle, Tile


def get_bounds(
    rectangles: Collection[Rectangle],
) -> Rectangle:
    """
    Finds the bounding rectangle for a collection of rectangles
    """
    x_max = y_max = -maxsize - 1
    x_min = y_min = maxsize
    level = rectangles[0]["level"]

    for rect in rectangles:
        if rect["level"] != level:
            raise ValueError("Rectangles must have the same level")
        x, y = rect["upper_left"]
        x_min = min(x, x_min)
        x_max = max(x + rect["width"], x_max)
        y_min = min(y, y_min)
        y_max = max(y + rect["height"], y_max)

    return Rectangle(upper_left=(x_min, y_min), width=x_max - x_min, height=y_max - y_min, level=level)


def combine_tiles(
    tiles: Collection[Tile],
) -> Tile:
    """
    Stitch tiles to an array of size original_height x original_width
    """
    rectangles = [tile["rect"] for tile in tiles]

    bounds = get_bounds(rectangles)
    x_min, y_min = bounds["upper_left"]
    combined_shape = (bounds["width"], bounds["height"])

    combined = np.zeros(combined_shape)
    normalization = np.zeros_like(combined)

    images = [tile["image"] for tile in tiles]
    for image, rect in zip(images, rectangles):
        x, y = rect["upper_left"]
        x_start = x - x_min
        x_end = x_start + rect["width"]
        y_start = y - y_min
        y_end = y_start + rect["height"]
        combined[x_start:x_end, y_start:y_end] += image
        normalization[x_start:x_end, y_start:y_end] += 1

    combined /= normalization

    return Tile(image=combined, rect=bounds)
