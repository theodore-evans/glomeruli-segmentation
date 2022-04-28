import numpy as np
from pytest import raises

from glomeruli_segmentation.data_classes import Rectangle, Tile
from glomeruli_segmentation.get_patches import get_patch_rectangles


def make_tile(rect: Rectangle):
    width, height = (rect.width, rect.height)
    image = np.arange(width * height, dtype=np.float32)
    image = image.reshape((width, height))
    image *= 255 / (width * height)
    return Tile(image=image, rect=rect)


def make_tile_getter(tile: Tile):
    def tile_getter(rect: Rectangle):
        x_start = rect.upper_left.x
        x_end = x_start + rect.width
        y_start = rect.upper_left.y
        y_end = y_start + rect.height
        return Tile(image=tile.image.copy()[x_start:x_end, y_start:y_end], rect=rect)

    return tile_getter


def test_returns_one_tile():
    rect = Rectangle(upper_left=(0, 0), width=10, height=10, level=0)
    rects = get_patch_rectangles(region=rect, window=(10, 10))

    assert next(rects) == rect
    with raises(StopIteration):
        next(rects)


def test_returns_two_tiles():
    left_rect = Rectangle(upper_left=(0, 0), width=10, height=10, level=0)
    right_rect = Rectangle(upper_left=(10, 0), width=10, height=10, level=0)
    combined_rect = Rectangle(upper_left=(0, 0), width=20, height=10, level=0)
    rects = get_patch_rectangles(region=combined_rect, window=(10, 10))

    for rect in left_rect, right_rect:
        assert next(rects) == rect
    with raises(StopIteration):
        next(rects)


def test_returns_four_identical_tiles():
    rect = Rectangle(upper_left=(0, 0), width=20, height=20, level=0)
    rects = get_patch_rectangles(region=rect, window=(10, 10))

    upper_lefts = [(0, 0), (10, 0), (0, 10), (10, 10)]
    correct_rects = [Rectangle(upper_left=upper_left, width=10, height=10, level=0) for upper_left in upper_lefts]
    for correct_rect in correct_rects:
        assert next(rects) == correct_rect
    with raises(StopIteration):
        next(rects)


def test_returns_four_non_identical_tiles():
    region = Rectangle(upper_left=(0, 0), width=20, height=20, level=0)

    rects = get_patch_rectangles(region=region, window=(10, 10))

    upper_lefts = [(0, 0), (10, 0), (0, 10), (10, 10)]
    correct_rects = [Rectangle(upper_left=upper_left, width=10, height=10, level=0) for upper_left in upper_lefts]

    for correct_rect in correct_rects:
        assert next(rects) == correct_rect
    with raises(StopIteration):
        next(rects)


def test_returns_overlapping_tiles_when_window_does_not_exactly_divide_region():
    region = Rectangle(upper_left=(0, 0), width=16, height=18, level=0)

    rects = get_patch_rectangles(region=region, window=(10, 10))

    upper_lefts = [(0, 0), (6, 0), (0, 8), (6, 8)]
    correct_rects = [Rectangle(upper_left=upper_left, width=10, height=10, level=0) for upper_left in upper_lefts]

    for correct_rect in correct_rects:
        assert next(rects) == correct_rect
    with raises(StopIteration):
        next(rects)


def test_returns_tiles_with_stride_no_equal_to_window_size():
    region = Rectangle(upper_left=(0, 0), width=18, height=16, level=0)

    rects = get_patch_rectangles(region=region, window=(10, 10), stride=(6, 6))

    upper_lefts = [(0, 0), (6, 0), (8, 0), (0, 6), (6, 6), (8, 6)]
    correct_rects = [Rectangle(upper_left=upper_left, width=10, height=10, level=0) for upper_left in upper_lefts]
    print(correct_rects)

    for correct_rect in correct_rects:
        assert next(rects) == correct_rect
    with raises(StopIteration):
        next(rects)


def test_handles_large_tiles():
    region = Rectangle(upper_left=(0, 0), width=2999, height=2011, level=0)
    rects = get_patch_rectangles(region=region, window=(1024, 1024), stride=(256, 256))

    rect_list = list(rect for rect in rects)
    assert len(rect_list) == 45
