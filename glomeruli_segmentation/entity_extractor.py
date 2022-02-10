from typing import List, Tuple, Union

import cv2 as cv
import numpy as np
from numpy import ndarray, uint8

from glomeruli_segmentation.data_classes import Tile, Vector2

_Contour = List[Tuple[int, int]]


def _threshold_to_binary_mask(image: ndarray, threshold: float = 0.5, positive_value: uint8 = uint8(255)) -> ndarray:
    thresholded_mask = np.zeros_like(image, dtype=uint8)
    _, thresholded_mask = cv.threshold(
        src=image, dst=thresholded_mask, thresh=threshold, maxval=positive_value, type=cv.THRESH_BINARY
    )
    return np.array(thresholded_mask, dtype=uint8)


def _find_contours(thresholded_mask: ndarray) -> List[_Contour]:
    contours, _ = cv.findContours(image=thresholded_mask, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
    return [contour.squeeze() for contour in contours]


def _get_contour_confidence(contour: _Contour, mask: ndarray, thresholded_mask: ndarray) -> float:
    y, x, h, w = cv.boundingRect(contour)
    bounded_mask = mask[x : x + w, y : y + h]
    bounded_binary_mask = thresholded_mask[x : x + w, y : y + h]
    pixel_confidences = bounded_mask[bounded_binary_mask != 0]
    mean_pixel_confidence = np.mean(pixel_confidences)
    return mean_pixel_confidence


def get_contours_from_mask(mask_tile: Tile, threshold: float = 0.5) -> Tuple[List[_Contour], List[float]]:
    mask = mask_tile.image
    thresholded_mask = _threshold_to_binary_mask(mask, threshold)
    contours = _find_contours(thresholded_mask)
    offset_contours = [contour + tuple(mask_tile.rect.upper_left) for contour in contours]
    confidences = [_get_contour_confidence(contour, mask, thresholded_mask) for contour in contours]
    return offset_contours, confidences
