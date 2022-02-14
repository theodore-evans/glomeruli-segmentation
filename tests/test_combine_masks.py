from typing import Collection, List

import numpy as np

from glomeruli_segmentation.data_classes import Rectangle, Tile
from glomeruli_segmentation.util.combine_masks import combine_masks

square = np.ones((10, 10))
rect1 = Rectangle(upper_left=(0, 0), width=10, height=10)
rect2 = Rectangle(upper_left=(10, 0), width=10, height=10)
rect3 = Rectangle(upper_left=(0, 10), width=10, height=10)
rect4 = Rectangle(upper_left=(10, 10), width=10, height=10)
non_overlapping_squares = [rect1, rect2, rect3, rect4]

square_tile = lambda rect: Tile(image=square, rect=rect)


def _split_image_into_tiles(original: np.ndarray, rectangles: List[Rectangle]) -> Collection[Tile]:
    tiles = []
    for rect in rectangles:
        x_start = rect.upper_left.x
        x_end = x_start + rect.width
        y_start = rect.upper_left.y
        y_end = y_start + rect.height
        tiles.append(Tile(image=original.copy()[y_start:y_end, x_start:x_end], rect=rect))
    return tiles


def test_combine_two_square_tiles_horizontally():
    combined = combine_masks((square_tile(rect1), square_tile(rect2)))
    assert combined.image.shape == (10, 20)


def test_combine_two_square_tiles_vertically():
    combined = combine_masks((square_tile(rect1), square_tile(rect3)))
    assert combined.image.shape == (20, 10)


def test_combine_four_square_tiles():
    combined = combine_masks((square_tile(rect1), square_tile(rect2), square_tile(rect3), square_tile(rect4)))
    assert combined.image.shape == (20, 20)


def test_recombine_square_tiles():
    original = np.arange(20 * 20).reshape(20, 20)
    tiles = _split_image_into_tiles(original, non_overlapping_squares)

    combined = combine_masks(tiles)
    assert np.array_equal(combined.image, original)


def test_recombine_non_square_tiles():
    original = np.arange(20 * 30).reshape(30, 20)
    rectangles = [
        Rectangle(upper_left=(0, 0), width=10, height=30),
        Rectangle(upper_left=(10, 0), width=10, height=30),
    ]
    tiles = _split_image_into_tiles(original, rectangles)

    combined = combine_masks(tiles)
    assert np.array_equal(combined.image, original)
    assert combined.rect == Rectangle(upper_left=(0, 0), width=20, height=30)


def test_combine_one_single_tile():
    original = np.arange(20 * 30).reshape(30, 20)
    rectangle = Rectangle(upper_left=(0, 0), width=20, height=30)

    combined = combine_masks([Tile(image=original, rect=rectangle)])
    assert np.array_equal(combined.image, original)
    assert combined.rect == rectangle


def test_recombine_with_non_zero_origin():
    original = np.arange(20 * 20).reshape(20, 20)
    tiles = _split_image_into_tiles(original, non_overlapping_squares)

    for tile in tiles:
        tile.rect.upper_left = tuple(x + 5 for x in tile.rect.upper_left)

    combined = combine_masks(tiles)
    assert np.array_equal(combined.image, original)
    assert combined.rect == Rectangle(upper_left=(5, 5), width=20, height=20)


def test_recombined_with_overlapping_tiles():
    shape = (15, 15)
    original = np.arange(shape[0] * shape[1]).reshape(shape)

    rectangles = [
        Rectangle(upper_left=(0, 0), width=10, height=10),
        Rectangle(upper_left=(5, 0), width=10, height=10),
        Rectangle(upper_left=(0, 5), width=10, height=10),
        Rectangle(upper_left=(5, 5), width=10, height=10),
    ]
    tiles = _split_image_into_tiles(original, rectangles)

    combined = combine_masks(tiles)
    assert np.array_equal(combined.image, original)
    assert combined.rect == Rectangle(upper_left=(0, 0), width=shape[0], height=shape[1])


def test_recombined_with_averaged_overlaps():
    shape = (10, 15)
    original = np.zeros(shape)
    original_rect = Rectangle(upper_left=(0, 0), width=shape[1], height=shape[0])
    rectangles = [
        Rectangle(upper_left=(0, 0), width=10, height=10),
        Rectangle(upper_left=(5, 0), width=10, height=10),
    ]
    tiles = _split_image_into_tiles(original, rectangles)
    tiles[0].image += 100
    combined = combine_masks(tiles)
    original_blended = np.concatenate([np.full((10, 5), 100), np.full((10, 5), 50), np.full((10, 5), 0)], axis=1)

    assert np.array_equal(combined.image, original_blended)
    assert combined.rect == original_rect


def test_recombine_large_tile_with_image_like_array():
    shape = (1024, 5000)
    original = np.arange(shape[0] * shape[1]).reshape(shape)
    rectangles = [
        Rectangle(upper_left=(0, 0), width=1024, height=1024),
        Rectangle(upper_left=(1024, 0), width=1024, height=1024),
        Rectangle(upper_left=(2048, 0), width=1024, height=1024),
        Rectangle(upper_left=(3072, 0), width=1024, height=1024),
        Rectangle(upper_left=(3976, 0), width=1024, height=1024),
    ]
    tiles = _split_image_into_tiles(original, rectangles)

    combined = combine_masks(tiles)
    assert np.array_equal(combined.image, original)
    assert combined.rect == Rectangle(upper_left=[0, 0], width=shape[1], height=shape[0])
