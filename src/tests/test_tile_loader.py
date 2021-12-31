from itertools import product
from typing import Tuple

import numpy as np
import pytest
from app import tile_loader
from app.data_types import Rectangle, Tile
from app.tile_loader import get_tile_loader

from util.combine_tiles import combine_tiles, get_bounds


def _make_tile(rect: Rectangle):
    width, height = (rect["width"], rect["height"])
    image = np.arange(width * height).reshape((width, height))
    return Tile(image=image, rect=rect)


def _make_tile_getter(tile: Tile):
    def tile_getter(rect: Rectangle):
        x_start = rect["upper_left"][0]
        x_end = x_start + rect["width"]
        y_start = rect["upper_left"][1]
        y_end = y_start + rect["height"]
        return Tile(image=tile["image"].copy()[x_start:x_end, y_start:y_end], rect=rect)

    return tile_getter


def test_returns_one_tile():
    rect = Rectangle(upper_left=(0, 0), width=10, height=10, level=0)
    tile = _make_tile(rect)
    tiles = get_tile_loader(get_tile=_make_tile, region=rect, window=(10, 10))
    first = next(tiles)
    assert np.array_equal(first["image"], tile["image"])
    assert first["rect"] == rect
    with pytest.raises(StopIteration):
        next(tiles)


def test_returns_two_tiles():
    left_rect = Rectangle(upper_left=(0, 0), width=10, height=10, level=0)
    right_rect = Rectangle(upper_left=(10, 0), width=10, height=10, level=0)
    left_tile = _make_tile(left_rect)
    right_tile = _make_tile(right_rect)
    combined_tile = combine_tiles([left_tile, right_tile])
    tiles = get_tile_loader(get_tile=_make_tile, region=combined_tile["rect"], window=(10, 10))

    first = next(tiles)
    print(f"{first['image'].shape=}, {left_tile['image'].shape=}")
    assert first["image"].shape == left_tile["image"].shape
    assert first["rect"] == left_rect
    assert np.array_equal(first["image"], left_tile["image"])

    second = next(tiles)
    assert second["rect"] == right_rect
    assert np.array_equal(second["image"], right_tile["image"])
    with pytest.raises(StopIteration):
        next(tiles)


def test_returns_four_identical_tiles():
    rect = Rectangle(upper_left=(0, 0), width=20, height=20, level=0)
    tiles = get_tile_loader(get_tile=_make_tile, region=rect, window=(10, 10))

    upper_lefts = [(0, 0), (10, 0), (0, 10), (10, 10)]
    rects = [Rectangle(upper_left=upper_left, width=10, height=10, level=0) for upper_left in upper_lefts]
    for rect in rects:
        desired_tile = _make_tile(rect)
        next_tile = next(tiles)
        assert next_tile["rect"] == desired_tile["rect"]
        assert np.array_equal(next_tile["image"], desired_tile["image"])
    with pytest.raises(StopIteration):
        next(tiles)


def test_returns_four_non_identical_tiles():
    region = Rectangle(upper_left=(0, 0), width=20, height=20, level=0)
    original_tile = _make_tile(region)

    tile_getter = _make_tile_getter(original_tile)
    tiles = get_tile_loader(get_tile=tile_getter, region=region, window=(10, 10))

    upper_lefts = [(0, 0), (10, 0), (0, 10), (10, 10)]
    rects = [Rectangle(upper_left=upper_left, width=10, height=10, level=0) for upper_left in upper_lefts]
    for rect in rects:
        desired_tile = tile_getter(rect)
        next_tile = next(tiles)
        assert next_tile["rect"] == desired_tile["rect"]
        assert np.array_equal(next_tile["image"], desired_tile["image"])
    with pytest.raises(StopIteration):
        next(tiles)


def test_returns_overlapping_tiles_when_window_does_not_exactly_divide_region():
    region = Rectangle(upper_left=(0, 0), width=16, height=18, level=0)
    original_tile = _make_tile(region)

    tile_getter = _make_tile_getter(original_tile)
    tiles = get_tile_loader(get_tile=tile_getter, region=region, window=(10, 10))

    upper_lefts = [(0, 0), (6, 0), (0, 8), (6, 8)]
    rects = [Rectangle(upper_left=upper_left, width=10, height=10, level=0) for upper_left in upper_lefts]
    for rect in rects:
        desired_tile = tile_getter(rect)
        next_tile = next(tiles)
        assert next_tile["rect"] == desired_tile["rect"]
        assert np.array_equal(next_tile["image"], desired_tile["image"])
    with pytest.raises(StopIteration):
        next(tiles)


def test_returns_tiles_with_stride_no_equal_to_window_size():
    region = Rectangle(upper_left=(0, 0), width=18, height=16, level=0)
    original_tile = _make_tile(region)

    tile_getter = _make_tile_getter(original_tile)
    tiles = get_tile_loader(get_tile=tile_getter, region=region, window=(10, 10), stride=(6, 6))

    upper_lefts = [(0, 0), (6, 0), (8, 0), (0, 6), (6, 6), (8, 6)]
    rects = [Rectangle(upper_left=upper_left, width=10, height=10, level=0) for upper_left in upper_lefts]
    print(rects)
    for rect in rects:
        desired_tile = tile_getter(rect)
        next_tile = next(tiles)
        print(next_tile["rect"]["upper_left"])
        assert next_tile["rect"] == desired_tile["rect"]
        assert np.array_equal(next_tile["image"], desired_tile["image"])
    with pytest.raises(StopIteration):
        next(tiles)

def test_handles_large_tiles():
    region = Rectangle(upper_left=(0, 0), width=2999, height=2011, level=0)
    original_tile = _make_tile(region)
    tile_getter = _make_tile_getter(original_tile)
    tiles = get_tile_loader(get_tile=tile_getter, region=region, window=(1024, 1024), stride=(256, 256))
    
    tile_list = list(tile for tile in tiles)
    assert get_bounds([tile['rect'] for tile in tile_list]) == region
    assert len(tile_list) == 45
    
