from typing import List
from numpy import ndarray

class EntityExtractor:
    def __init__(self):
        pass
    
    def extract_contours(self, segmentation_mask: ndarray) -> List:
        """
        Extract coutours from segmentation mask (Not implemented placeholder)
        """
        return [None] * 42
    
    def count_entities(self, contours: List) -> int:
        """
        Count the number of individual entities in a list of contour (Not implemented placeholder)
        """
        return len(contours)