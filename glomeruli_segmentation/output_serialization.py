from typing import Any, Callable, Dict, List, Tuple
from uuid import UUID

from glomeruli_segmentation.data_classes import Rectangle, Wsi


def create_annotation_collection(contours: List[Tuple[int, int]], slide: Wsi, roi: Rectangle) -> Dict[str, Any]:

    npp = slide.pixel_size_nm.x
    num_levels = len(slide.levels)
    num_contours = len(contours)

    items: List[dict] = []

    for i, contour in enumerate(contours):
        item = {
            "name": "Glomerulus",
            "description": f"Glomerulus {i+1}/{num_contours}",
            "type": "polygon",
            "reference_id": slide.id,
            "reference_type": "wsi",
            "coordinates": contour,
            "npp_created": npp,
            "npp_viewing": [npp, npp * 2 ** num_levels],
        }
        items.append(item)

    return {
        "item_type": "polygon",
        "items": items,
        "reference_id": roi.id,
        "reference_type": "annotation",
    }


def classify_annotations(
    confidences: List[float], condition: Callable[..., bool], class_if_true: str, class_if_false: str
) -> List[str]:

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


def link_result_details(
    annotations: Dict[str, Any], confidences: List[float], classifications: List[str]
) -> Dict[str, Any]:

    confidence_collection = {"item_type": "float", "items": []}
    classification_collection = {"item_type": "class", "items": []}

    for annotation, confidence, classification in zip(annotations["items"], confidences, classifications):
        try:
            UUID(annotation["id"])
        except (KeyError, ValueError) as invalid_id:
            raise ValueError(
                f"{annotation['name']} id is missing or invalid, POST annotation results before linking details"
            ) from invalid_id

        confidence_collection["items"].append(
            {
                "name": "Confidence",
                "type": "float",
                "value": confidence,
                "reference_id": annotation["id"],
                "reference_type": "annotation",
            }
        )

        classification_collection["items"].append(
            {
                "value": classification,
                "reference_id": annotation["id"],
                "reference_type": "annotation",
            }
        )

    return confidence_collection, classification_collection
