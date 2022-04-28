from typing import Callable, List, Tuple

import cv2 as cv
import numpy as np
from numpy import ndarray, uint8

from glomeruli_segmentation.config import Config
from glomeruli_segmentation.data_classes import Mask

_Contour = List[List[int]]


def _threshold_to_binary_mask(image: ndarray, threshold: float = 0.5, positive_value: uint8 = uint8(255)) -> ndarray:
    """
    Thresholds an image to a binary mask.

    :param image: A single-channel image.
    :param threshold: The threshold to apply.
    :param positive_value: The value to assign to pixels that are above the threshold.
    :return: A binary mask as ndarray.
    """
    thresholded_image = np.zeros_like(image, dtype=uint8)
    _, thresholded_image = cv.threshold(
        src=image, dst=thresholded_image, thresh=threshold, maxval=positive_value, type=cv.THRESH_BINARY
    )
    return np.array(thresholded_image, dtype=uint8)


def _find_contours(binary_mask_image: ndarray) -> List[_Contour]:
    """
    Finds contours in a single-channel, binary valued image.

    :param binary_mask_image: A single-channel, binary valued image.
    :return: A list of contours.
    """
    contours, _ = cv.findContours(image=binary_mask_image, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
    return [contour.squeeze() for contour in contours]


def _get_contour_confidence(
    contour: _Contour, mask: ndarray, thresholded_mask: ndarray, num_of_decimals: int = 3
) -> float:
    """
    Gets the confidence of a contour.

    :param contour: A contour.
    :param mask: The mask from which the contour was extracted.
    :param thresholded_mask: The thresholded mask from which the contour was extracted.
    :param num_of_decimals: The number of decimals to which to round the confidence.
    :return: The confidence of the contour.
    """
    y, x, h, w = cv.boundingRect(contour)
    bounded_mask = mask[x : x + w, y : y + h]
    bounded_binary_mask = thresholded_mask[x : x + w, y : y + h]
    pixel_confidences = bounded_mask[bounded_binary_mask != 0]
    mean_pixel_confidence = np.mean(pixel_confidences)
    return round(mean_pixel_confidence, num_of_decimals)


def _get_contours_from_mask(mask: Mask, threshold: float = 0.5) -> Tuple[List[_Contour], List[float]]:
    """
    Gets contours from a mask.

    :param mask: A mask with mask.image.shape == (height, width)
    :param threshold: The threshold to use to produce a binary mask.
    :return: A tuple containing:
        - A list of contours.
        - A list of confidences.
    """
    thresholded_mask_image = _threshold_to_binary_mask(mask.image, threshold)
    contours = _find_contours(thresholded_mask_image)
    offset_contours = [(mask.rect.upper_left + contour).tolist() for contour in contours]
    confidences = [_get_contour_confidence(contour, mask.image, thresholded_mask_image) for contour in contours]
    return offset_contours, confidences


def _classify_annotations(
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

    assert num_positive_classifications + num_negative_classifications == len(confidences)
    return classifications, num_positive_classifications, num_negative_classifications


def get_results_from_mask(segmentation_mask: Mask, config: Config) -> dict:
    """
    Gets the application results from segmentation mask.

    :param config: A Config object containing the configuration parameters 'anomaly_confidence_threshold', 'anomaly_class_name', 'normal_class_name', 'binary_threshold'.
    :param segmentation_mask: A segmentation mask of type Mask.
    :return: A dictionary containing:
        - A list of contours.
        - A list of confidences.
        - A list of classifications.
        - The number of positive classifications.
    """
    contours, confidences = _get_contours_from_mask(segmentation_mask, config.binary_threshold)
    classifications, count, _ = _classify_annotations(
        confidences,
        condition=lambda x: x > config.anomaly_confidence_threshold,
        class_if_true=config.normal_class,
        class_if_false=config.anomaly_class,
    )

    return {"contours": contours, "confidences": confidences, "classifications": classifications, "count": count}
