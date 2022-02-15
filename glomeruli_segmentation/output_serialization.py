from typing import Any, Callable, Dict, List, Tuple

from glomeruli_segmentation.data_classes import Rectangle, Wsi


def contours_to_collection(contours: List[Tuple[int, int]], slide: Wsi, roi: Rectangle) -> Dict[str, Any]:

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
            "coordinates": contour,  # Always use WSI base level coordinates
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


def confidences_to_collection(annotations: Dict[str, Any], confidence_values: List[float]) -> Dict[str, Any]:

    items = []
    for annotation, confidence_value in zip(annotations["items"], confidence_values):
        if annotation["id"] is None:
            raise KeyError(f"{annotation['name']} has no id, POST results before assigning confidences")
        items.append(
            {
                "name": "confidence score",
                "type": "float",
                "value": confidence_value,
                "reference_id": annotation["id"],
                "reference_type": "annotation",
            }
        )

    return {"item_type": "float", "items": items}


def classifications_to_collection(
    annotations: Dict[str, Any],
    confidence_values: List[float],
    pos_condition: Callable[..., bool],
    pos_class: str,
    neg_class: str,
) -> Dict[str, Any]:

    items = []
    for annotation, confidence_value in zip(annotations["items"], confidence_values):
        if annotation["id"] is None:
            raise KeyError(f"{annotation['name']} has no id, POST results before assigning classes")
        classification = pos_class if pos_condition(confidence_value) else neg_class
        items.append(
            {
                "value": classification,
                "reference_id": annotation["id"],
                "reference_type": "annotation",
            }
        )

    return {"item_type": "class", "items": items}
