from typing import List
from numpy import ndarray
import cv2 as cv
import numpy as np


class EntityExtractor:
    def __init__(self,  upper_left: List[int], threshold: float = 0.5):
        self.threshold = threshold
        self.upper_left = upper_left

    # Need to pass one channel mask here
    def treshold_mask(self, segmentation_mask: ndarray) -> ndarray:
        threshold_value = int(self.threshold*255)
        _, thresh = cv.threshold(segmentation_mask, threshold_value, 255, 0)
        return thresh

    def extract_contours(self, segmentation_mask: ndarray) -> List:
        """
        Extract coutours from segmentation mask (Not implemented placeholder)
        """
        tresholded_mask = self.treshold_mask(segmentation_mask)
        contours, _ = cv.findContours(
            tresholded_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = [(contour.squeeze() + self.upper_left).tolist()
                    for contour in contours]
        return contours

    def count_entities(self, contours: List) -> int:
        """
        Count the number of individual entities in a list of contour (Not implemented placeholder)
        """
        return len(contours)
