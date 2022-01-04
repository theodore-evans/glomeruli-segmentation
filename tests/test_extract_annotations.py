from typing import Collection
import cv2

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from cv2 import contourArea
from glomeruli_segmentation.data_classes import Rectangle, Tile, Vector2
from glomeruli_segmentation.entity_extractor import get_contours_from_mask
from glomeruli_segmentation.util.combine_masks import combine_masks



def _make_circle(side: int, radius: int):
    xx, yy = np.mgrid[:side, :side]
    circle = (xx - side / 2) ** 2 + (yy - side / 2) ** 2
    return np.array((circle < radius ** 2), dtype=float)


def test_returns_empty_list_of_contours_for_blank_tile():
    shape = (256, 256, 1)
    rect = Rectangle(upper_left=Vector2(0, 0), width=shape[0], height=shape[1])
    blank_image = np.zeros((shape))
    blank_tile = Tile(image=blank_image, rect=rect)
    empty_contours, _ = get_contours_from_mask(blank_tile)

    assert isinstance(empty_contours, Collection)
    assert len(empty_contours) == 0


def test_returns_one_contour_for_circular_mask():
    shape = (256, 256, 1)
    rect = Rectangle(upper_left=Vector2(0, 0), width=shape[0], height=shape[1])
    circular_mask = _make_circle(shape[0], radius=100)
    circular_mask_tile = Tile(image=circular_mask, rect=rect)
    contours, _ = get_contours_from_mask(circular_mask_tile)

    assert isinstance(contours, Collection)
    assert isinstance(contours[0], Collection)
    assert len(contours[0][0]) == 2
    assert len(contours) == 1
    
def test_returns_two_contours_for_two_circular_masks():
    shape = (256, 256, 1)
    rect_left = Rectangle(upper_left=Vector2(0, 0), width=shape[0], height=shape[1])
    rect_right = Rectangle(upper_left=Vector2(shape[0], 0), width=shape[0], height=shape[1])
    circular_mask = _make_circle(shape[0], radius=100)
    tiles = [Tile(image=circular_mask, rect=rect) for rect in (rect_left, rect_right)]
    contours, _ = get_contours_from_mask(combine_masks(tiles))

    assert isinstance(contours, Collection)
    assert isinstance(contours[0], Collection)
    assert len(contours[0][0]) == 2
    assert len(contours) == 2

def test_returns_one_contour_for_blurred_circular_mask():
    shape = (256, 256, 1)
    rect = Rectangle(upper_left=Vector2(0, 0), width=shape[0], height=shape[1])
    circular_mask = _make_circle(shape[0], radius=100)
    blurred_circular_mask = gaussian_filter(circular_mask, sigma=(10, 10))
    blurred_circular_mask_tile = Tile(image=blurred_circular_mask, rect=rect)
    contours, _ = get_contours_from_mask(blurred_circular_mask_tile)

    assert isinstance(contours, Collection)
    assert isinstance(contours[0], Collection)
    assert len(contours[0][0]) == 2
    assert len(contours) == 1

def test_returns_different_contour_for_different_thresholds():
    shape = (256, 256, 1)
    rect = Rectangle(upper_left=Vector2(0, 0), width=shape[0], height=shape[1])
    circular_mask = _make_circle(shape[0], radius=100)
    blurred_circular_mask = gaussian_filter(circular_mask, sigma=(10, 10))
    blurred_circular_mask_tile = Tile(image=blurred_circular_mask, rect=rect)

    contours_low, _ = get_contours_from_mask(blurred_circular_mask_tile, threshold=0.25)
    contours_high, _ = get_contours_from_mask(blurred_circular_mask_tile, threshold=0.75)

    assert len(contours_high) == 1 
    assert len(contours_low) == 1
    assert len(contours_high[0]) != len(contours_low[0]) or not np.allclose(contours_high, contours_low)
    assert contourArea(contours_high[0]) < contourArea(contours_low[0])

def test_returns_shifted_contours_for_shifted_tile():
    shape = (256, 256, 1)
    rect = Rectangle(upper_left=Vector2(200, 200), width=shape[0], height=shape[1])
    circular_mask = _make_circle(shape[0], radius=100)
    blurred_circular_mask = gaussian_filter(circular_mask, sigma=(10, 10))
    blurred_circular_mask_tile = Tile(image=blurred_circular_mask, rect=rect)
    contours, _ = get_contours_from_mask(blurred_circular_mask_tile)

    assert isinstance(contours, Collection)
    assert isinstance(contours[0], Collection)
    assert len(contours) == 1
    for contour in contours:
        for coordinate in contour:
            x,y = coordinate
            assert x > 200 and y > 200

def test_returns_one_confidence_per_annotation():
    shape = (256, 256, 1)
    rect_left = Rectangle(upper_left=Vector2(0, 0), width=shape[0], height=shape[1])
    rect_right = Rectangle(upper_left=Vector2(shape[0], 0), width=shape[0], height=shape[1])
    circular_mask = _make_circle(shape[0], radius=100)
    tiles = [Tile(image=circular_mask, rect=rect) for rect in (rect_left, rect_right)]
    contours, confidences = get_contours_from_mask(combine_masks(tiles))
    
    assert len(contours) == len(confidences)
    assert confidences[0] == confidences[1]
    print(confidences)
    assert all(confidence >= 0 and confidence <= 1 for confidence in confidences)
    assert False