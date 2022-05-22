from typing import Collection

import numpy as np
from cv2 import contourArea
from scipy.ndimage import gaussian_filter

from glomeruli_segmentation.data_classes import Rect, Tile
from glomeruli_segmentation.extract_results import find_contours
from glomeruli_segmentation.util import combine_masks


def _threshold(image, t):
    thresholded = np.array(image, copy=True)
    thresholded[thresholded < t] = 0
    return thresholded


def _make_circle(side: int, radius: int):
    xx, yy = np.mgrid[:side, :side]
    circle = (xx - side / 2) ** 2 + (yy - side / 2) ** 2
    return np.array((circle < radius**2), dtype=float)


def test_returns_empty_list_of_contours_for_blank_tile():
    shape = (256, 256, 1)
    rect = Rect(upper_left=(0, 0), width=shape[0], height=shape[1])
    blank_image = np.zeros((shape))
    blank_tile = Tile(image=blank_image, rect=rect)
    empty_contours = find_contours(blank_tile)

    assert isinstance(empty_contours, Collection)
    assert len(empty_contours) == 0


def test_returns_one_contour_for_circular_mask():
    shape = (256, 256, 1)
    rect = Rect(upper_left=(0, 0), width=shape[0], height=shape[1])
    circular_mask = _make_circle(shape[0], radius=100)
    circular_mask_tile = Tile(image=circular_mask, rect=rect)
    contours = find_contours(circular_mask_tile)

    assert isinstance(contours, Collection)
    assert isinstance(contours[0], Collection)
    assert len(contours[0]) > 10
    assert len(contours[0][0]) == 2
    assert len(contours) == 1


def test_returns_two_contours_for_two_circular_masks():
    shape = (256, 256, 1)
    rect_left = Rect(upper_left=(0, 0), width=shape[0], height=shape[1])
    rect_right = Rect(upper_left=(shape[0], 0), width=shape[0], height=shape[1])
    circular_mask = _make_circle(shape[0], radius=100)
    tiles = [Tile(image=circular_mask, rect=rect) for rect in (rect_left, rect_right)]
    contours = find_contours(combine_masks(tiles))

    assert isinstance(contours, Collection)
    assert isinstance(contours[0], Collection)
    assert len(contours[0]) > 10
    assert len(contours[0][0]) == 2
    assert len(contours) == 2


def test_returns_one_contour_for_blurred_circular_mask():
    shape = (256, 256, 1)
    rect = Rect(upper_left=(0, 0), width=shape[0], height=shape[1])
    circular_mask = _make_circle(shape[0], radius=100)
    blurred_circular_mask = gaussian_filter(circular_mask, sigma=(10, 10))
    blurred_circular_mask_tile = Tile(image=blurred_circular_mask, rect=rect)
    contours = find_contours(blurred_circular_mask_tile)

    assert isinstance(contours, Collection)
    assert isinstance(contours[0], Collection)
    assert len(contours[0]) > 10
    assert len(contours[0][0]) == 2
    assert len(contours) == 1


def test_returns_different_contour_for_different_thresholds():
    shape = (256, 256, 1)
    rect = Rect(upper_left=(0, 0), width=shape[0], height=shape[1])
    circular_mask = _make_circle(shape[0], radius=100)
    blurred_circular_mask = gaussian_filter(circular_mask, sigma=10)
    blurred_circular_mask_tile = Tile(image=blurred_circular_mask, rect=rect)

    low_image = _threshold(blurred_circular_mask_tile.image, 0.25)
    high_image = _threshold(blurred_circular_mask_tile.image, 0.75)
    low = Tile(low_image, rect)
    high = Tile(high_image, rect)

    contours_low = find_contours(low)
    contours_high = find_contours(high)

    assert len(contours_high) == 1
    assert len(contours_low) == 1
    assert len(contours_high[0]) > 10
    assert len(contours_high[0]) != len(contours_low[0]) or not np.allclose(contours_high, contours_low)
    print(type(contours_high[0]), contours_high[0])
    assert contourArea(np.array(contours_high[0])) < contourArea(np.array(contours_low[0]))
