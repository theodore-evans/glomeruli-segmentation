from dataclasses import dataclass
from typing import List

import cv2 as cv
import numpy as np
from numpy import ndarray


@dataclass
class ExtractedEntities:
    contours: List
    confidences: List

    def count(self):
        return len(self.contours)


class EntityExtractor:
    def __init__(self, upper_left: List[int], threshold: float = 0.5):
        self.threshold = threshold
        self.upper_left = upper_left

    def threshold_mask(self, segmentation_mask: ndarray) -> ndarray:
        threshold_value = int(self.threshold * 255)
        _, thresh = cv.threshold(segmentation_mask, threshold_value, 255, 0)
        return thresh

    def get_contour_confidence(
        self, segmentation_mask: ndarray, thresholded_mask: ndarray, contour: ndarray
    ) -> float:
        y, x, h, w = cv.boundingRect(contour)
        bounded_seg_mask = segmentation_mask[x : x + w, y : y + h]
        bounded_thresh_mask = thresholded_mask[x : x + w, y : y + h]
        pixel_confidences = bounded_seg_mask[bounded_thresh_mask != 0]
        mean_pixel_confidence = np.mean(pixel_confidences) / 255.0
        return mean_pixel_confidence

    def extract_entities(self, segmentation_mask: ndarray) -> ExtractedEntities:
        """
        Extract coutours and confidences from segmentation mask
        """
        thresholded_mask = self.threshold_mask(segmentation_mask)
        contours, _ = cv.findContours(thresholded_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        confidences = [
            self.get_contour_confidence(segmentation_mask, thresholded_mask, contour) for contour in contours
        ]

        contours = [(contour.squeeze() + self.upper_left).tolist() for contour in contours]

        return ExtractedEntities(contours, confidences)
