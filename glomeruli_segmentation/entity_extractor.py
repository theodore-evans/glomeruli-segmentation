import functools
from dataclasses import dataclass
from typing import List, Tuple

import cv2 as cv
import numpy as np
from numpy import ndarray, uint8

from glomeruli_segmentation.data_classes import Tile

_Contour = List[Tuple[int,int]]
    
def _threshold_to_binary_mask(image: ndarray, threshold: float, positive_value: uint8=255) -> ndarray:
    thresholded_mask = np.zeros_like(image, dtype=uint8)
    _, thresholded_mask = cv.threshold(src=image, dst=thresholded_mask, thresh=threshold, maxval=positive_value, type=cv.THRESH_BINARY)
    return np.array(thresholded_mask, dtype=uint8)

def _find_contours(thresholded_mask: ndarray)-> List[_Contour]:
    contours, _ = cv.findContours(image=thresholded_mask, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
    return [contour.squeeze() for contour in contours]
    
def _get_contour_confidence(contour: _Contour, mask: ndarray, thresholded_mask: ndarray)->float:
    y, x, h, w = cv.boundingRect(contour)
    bounded_mask = mask[x : x + w, y : y + h]
    bounded_binary_mask = thresholded_mask[x : x + w, y : y + h]
    pixel_confidences = bounded_mask[bounded_binary_mask != 0]
    mean_pixel_confidence = np.mean(pixel_confidences)
    return mean_pixel_confidence

def get_contours_from_mask(mask_tile: Tile, threshold: float = 0.5)-> Tuple[List[_Contour], List[float]]:
    mask = mask_tile.image
    thresholded_mask = _threshold_to_binary_mask(mask, threshold)
    contours = _find_contours(thresholded_mask)
    offset_contours = [contour + mask_tile.rect.upper_left for contour in contours]
    confidences = [_get_contour_confidence(contour, mask, thresholded_mask) for contour in contours]
    return offset_contours, confidences


# class EntityExtractor:
#     def __init__(self, origin: List[int], threshold: float = 0.5):
#         self.threshold = threshold
#         self.origin = origin

#     def threshold_mask(self, segmentation_mask: ndarray) -> ndarray:
#         threshold_value = int(self.threshold * 255)
#         _, thresh = cv.threshold(segmentation_mask, threshold_value, 255, 0)
#         return thresh

#     def get_contour_confidence(
#         self, segmentation_mask: ndarray, thresholded_mask: ndarray, contour: ndarray
#     ) -> float:
#         y, x, h, w = cv.boundingRect(contour)
#         bounded_seg_mask = segmentation_mask[x : x + w, y : y + h]
#         bounded_thresh_mask = thresholded_mask[x : x + w, y : y + h]
#         pixel_confidences = bounded_seg_mask[bounded_thresh_mask != 0]
#         mean_pixel_confidence = np.mean(pixel_confidences) / 255.0
#         return mean_pixel_confidence

#     def extract_from_mask(self, segmentation_mask: ndarray) -> ExtractedEntities:
#         """
#         Extract coutours and confidences from segmentation mask
#         """
#         thresholded_mask = self.threshold_mask(segmentation_mask)
#         contours, _ = cv.findContours(thresholded_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#         confidences = [
#             self.get_contour_confidence(segmentation_mask, thresholded_mask, contour) for contour in contours
#         ]

#         contours = [(contour.squeeze() + self.origin).tolist() for contour in contours]

#         return ExtractedEntities(contours, confidences)
