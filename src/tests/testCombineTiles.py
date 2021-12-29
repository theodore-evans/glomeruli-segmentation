from typing import List

import numpy as np
from app.data_types import Rectangle, Tile
from util.combine_tiles import combine_tiles

square = np.ones((10, 10))
rect1 = Rectangle(upper_left=(0, 0), width=10, height=10, level=0)
rect2 = Rectangle(upper_left=(10, 0), width=10, height=10, level=0)
rect3 = Rectangle(upper_left=(0, 10), width=10, height=10, level=0)
rect4 = Rectangle(upper_left=(10, 10), width=10, height=10, level=0)
non_overlapping_squares = [rect1, rect2, rect3, rect4]

square_tile = lambda rect: Tile(image=square, rect=rect)


def _split_image_into_tiles(original: np.ndarray, rectangles: List[Rectangle]):
    tiles = []
    for rect in rectangles:
        x_start = rect["upper_left"][0]
        x_end = x_start + rect["width"]
        y_start = rect["upper_left"][1]
        y_end = y_start + rect["height"]
        tiles.append(Tile(image=original.copy()[x_start:x_end, y_start:y_end], rect=rect))
    return tiles


def test_combine_two_square_tiles_horizontally():
    combined = combine_tiles((square_tile(rect1), square_tile(rect2)))
    assert combined["image"].shape == (20, 10)


def test_combine_two_square_tiles_vertically():
    combined = combine_tiles((square_tile(rect1), square_tile(rect3)))
    assert combined["image"].shape == (10, 20)


def test_combine_four_square_tiles():
    combined = combine_tiles((square_tile(rect1), square_tile(rect2), square_tile(rect3), square_tile(rect4)))
    assert combined["image"].shape == (20, 20)


def test_recombine_square_tiles():
    elements = [i for i in range(0, 20 * 20)]
    original = np.reshape(elements, (20, 20))
    tiles = _split_image_into_tiles(original, non_overlapping_squares)

    combined = combine_tiles(tiles)
    assert np.array_equal(combined["image"], original)


def test_recombine_non_square_tiles():
    elements = [i for i in range(0, 20 * 30)]
    original = np.reshape(elements, (20, 30))
    rectangles = [
        Rectangle(upper_left=(0, 0), width=10, height=30, level=0),
        Rectangle(upper_left=(10, 0), width=10, height=30, level=0),
    ]
    tiles = _split_image_into_tiles(original, rectangles)

    combined = combine_tiles(tiles)
    assert np.array_equal(combined["image"], original)
    assert combined["rect"] == Rectangle(upper_left=(0, 0), width=20, height=30, level=0)


def test_combine_one_single_tile():
    elements = [i for i in range(0, 20 * 30)]
    original = np.reshape(elements, (20, 30))
    rectangle = Rectangle(upper_left=(0, 0), width=20, height=30, level=0)

    combined = combine_tiles([Tile(image=original, rect=rectangle)])
    assert np.array_equal(combined["image"], original)
    assert combined["rect"] == rectangle


def test_recombine_with_non_zero_origin():
    elements = [i for i in range(0, 20 * 20)]
    original = np.reshape(elements, (20, 20))
    tiles = _split_image_into_tiles(original, non_overlapping_squares)

    for tile in tiles:
        tile["rect"]["upper_left"] = tuple(x + 5 for x in tile["rect"]["upper_left"])

    combined = combine_tiles(tiles)
    assert np.array_equal(combined["image"], original)
    assert combined["rect"] == Rectangle(upper_left=(5, 5), width=20, height=20, level=0)


def test_recombined_with_overlapping_tiles():
    shape = (15, 15)
    elements = [i for i in range(0, shape[0] * shape[1])]
    original = np.reshape(elements, shape)

    rectangles = [
        Rectangle(upper_left=(0, 0), width=10, height=10, level=0),
        Rectangle(upper_left=(5, 0), width=10, height=10, level=0),
        Rectangle(upper_left=(0, 5), width=10, height=10, level=0),
        Rectangle(upper_left=(5, 5), width=10, height=10, level=0),
    ]
    tiles = _split_image_into_tiles(original, rectangles)

    combined = combine_tiles(tiles)
    assert np.array_equal(combined["image"], original)
    assert combined["rect"] == Rectangle(upper_left=(0, 0), width=shape[0], height=shape[1], level=0)


def test_recombined_with_averaged_overlaps():
    shape = (15, 10)
    original = np.zeros(shape)
    original_rect = Rectangle(upper_left=(0, 0), width=shape[0], height=shape[1], level=0)
    rectangles = [
        Rectangle(upper_left=(0, 0), width=10, height=10, level=0),
        Rectangle(upper_left=(5, 0), width=10, height=10, level=0),
    ]
    tiles = _split_image_into_tiles(original, rectangles)
    tiles[0]["image"] += 100
    print(tiles)
    combined = combine_tiles(tiles)
    original_blended = np.concatenate([np.full((5, 10), 100), np.full((5, 10), 50), np.full((5, 10), 0)])
    assert np.array_equal(combined["image"], original_blended)
    assert combined["rect"] == original_rect
