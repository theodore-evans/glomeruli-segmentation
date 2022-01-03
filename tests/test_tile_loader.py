import numpy as np
import pytest

from glomeruli_segmentation.data_classes import Rectangle, Tile
from glomeruli_segmentation.tile_loader import get_tile_loader
from glomeruli_segmentation.util.combine_masks import combine_masks, get_bounds


def _make_tile(rect: Rectangle):
    width, height = (rect.width, rect.height)
    image = np.arange(width * height).reshape((width, height))
    return Tile(image=image, rect=rect)


def _make_tile_getter(tile: Tile):
    def tile_getter(rect: Rectangle):
        x_start = rect.upper_left.x
        x_end = x_start + rect.width
        y_start = rect.upper_left.y
        y_end = y_start + rect.height
        return Tile(image=tile.image.copy()[x_start:x_end, y_start:y_end], rect=rect)

    return tile_getter


def test_returns_one_tile():
    rect = Rectangle(upper_left=(0, 0), width=10, height=10, level=0)
    tile = _make_tile(rect)
    tiles = get_tile_loader(get_tile=_make_tile, region=rect, window=(10, 10))

    assert next(tiles) == tile
    with pytest.raises(StopIteration):
        next(tiles)


def test_returns_two_tiles():
    left_rect = Rectangle(upper_left=(0, 0), width=10, height=10, level=0)
    right_rect = Rectangle(upper_left=(10, 0), width=10, height=10, level=0)
    left_tile = _make_tile(left_rect)
    right_tile = _make_tile(right_rect)
    combined_tile = combine_masks([left_tile, right_tile])
    tiles = get_tile_loader(get_tile=_make_tile, region=combined_tile.rect, window=(10, 10))

    for tile in left_tile, right_tile:
        assert next(tiles) == tile
    with pytest.raises(StopIteration):
        next(tiles)


def test_returns_four_identical_tiles():
    rect = Rectangle(upper_left=(0, 0), width=20, height=20, level=0)
    tiles = get_tile_loader(get_tile=_make_tile, region=rect, window=(10, 10))

    upper_lefts = [(0, 0), (10, 0), (0, 10), (10, 10)]
    rects = [Rectangle(upper_left=upper_left, width=10, height=10, level=0) for upper_left in upper_lefts]
    for rect in rects:
        assert next(tiles) == _make_tile(rect)
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
        assert next(tiles) == tile_getter(rect)
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
        assert next(tiles) == tile_getter(rect)
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
        assert next(tiles) == tile_getter(rect)
    with pytest.raises(StopIteration):
        next(tiles)


def test_handles_large_tiles():
    region = Rectangle(upper_left=(0, 0), width=2999, height=2011, level=0)
    original_tile = _make_tile(region)
    tile_getter = _make_tile_getter(original_tile)
    tiles = get_tile_loader(get_tile=tile_getter, region=region, window=(1024, 1024), stride=(256, 256))

    tile_list = list(tile for tile in tiles)
    assert get_bounds([tile.rect for tile in tile_list]) == region
    assert len(tile_list) == 45
