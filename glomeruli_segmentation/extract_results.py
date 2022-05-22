from typing import Callable, List, Tuple

import cv2 as cv
import numpy as np
import shapely.geometry as sg
import torch

from glomeruli_segmentation.data_classes import Mask

_Contour = List[List[int]]


def find_contours(mask: Mask) -> List[_Contour]:
    """
    Finds contours in a single-channel, binary valued image.

    :param binary_mask_image: A single-channel, binary valued image.
    :return: A list of contours.
    """
    if mask.image.dtype in (np.half, np.single, np.double, torch.half, torch.float, torch.double):
        image = np.array(mask.image[:] * 255, dtype=np.uint8)
    image = image.squeeze()
    contours, _ = cv.findContours(
        image=image, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE, offset=mask.rect.upper_left
    )
    return [contour.squeeze() for contour in contours]


def classify_annotations(
    confidences: List[float], condition: Callable[..., bool], class_if_true: str, class_if_false: str
) -> Tuple[List[str], int, int]:
    """
    Classifies annotations based on a condition.

    :param confidences: A list of confidences.
    :param condition: A function that takes a confidence and returns a boolean.
    :param class_if_true: The class to assign to the annotations that satisfy the condition.
    :param class_if_false: The class to assign to the annotations that do not satisfy the condition.
    :return: A tuple containing:
        - A list of classifications.
        - The number of positive classifications.
        - The number of negative classifications.
    """

    classifications = []
    num_positive_classifications = 0
    num_negative_classifications = 0

    for confidence in confidences:
        if condition(confidence):
            classification = class_if_true
            num_positive_classifications += 1
        else:
            classification = class_if_false
            num_negative_classifications += 1
        classifications.append(classification)

    return classifications, num_positive_classifications, num_negative_classifications


def merge_overlapping_polygons(polygons) -> List[sg.Polygon]:

    polygons = [sg.Polygon(polygon).buffer(0) for polygon in polygons if len(polygon) > 2]
    merged_polygon = polygons.pop()
    for polygon in polygons:
        merged_polygon = merged_polygon.union(polygon)
    return merged_polygon


def average_values_in_polygon(segmentation: Mask, polygon: sg.Polygon) -> float:

    offset = segmentation.rect.upper_left
    x_min, y_min, x_max, y_max = polygon.bounds

    x_min, x_max = int(x_min) - offset[0], int(x_max) - offset[0]
    y_min, y_max = int(y_min) - offset[1], int(y_max) - offset[1]

    values = segmentation.image[y_min:y_max, x_min:x_max]

    average_value = np.nanmean(values[values > 0])
    return average_value if not np.isnan(average_value) else 0.0
