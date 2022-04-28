from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID

from glomeruli_segmentation.data_classes import Rectangle, Wsi

Coordinate = Tuple[int, int]


def create_annotation_collection(
    name: str,
    slide: Wsi,
    roi: Rectangle,
    annotation_type: str,
    values: Union[Coordinate, List[Coordinate]],
    visible_levels: int = -1,
) -> Dict[str, Any]:
    """
    Create a collection of annotations for a given slide, roi, and annotation type.

    :param name: Name of the annotation collection
    :param slide: Wsi object
    :param roi: Rectangle object
    :param annotation_type: Type of annotation
    :param values: Coordinates of the annotation
    :param visible_levels: Number of levels to show in the annotation
    """

    npp = (slide.pixel_size_nm.x + slide.pixel_size_nm.y) / 2
    num_levels = len(slide.levels) if visible_levels < 0 else visible_levels
    num_contours = len(values)

    items: List[dict] = []

    for i, contour in enumerate(values):
        item = {
            "name": name,
            "description": f"{name} {i+1}/{num_contours}",
            "type": annotation_type,
            "reference_id": slide.id,
            "reference_type": "wsi",
            "coordinates": contour,
            "npp_created": npp,
            "npp_viewing": [npp, npp * 2**num_levels],
        }
        items.append(item)

    collection = {
        "item_type": annotation_type,
        "items": items,
        "reference_id": roi.id,
        "reference_type": "annotation",
    }

    return collection


def link_results_by_id(reference_result: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    """
    Link a set of results to a reference result by id.

    :param reference_result: Reference result
    :param results: List of results"""
    try:
        UUID(reference_result["id"])
    except (KeyError, ValueError) as invalid_id:
        raise ValueError(
            f"{reference_result['name']} id is missing or invalid, POST reference_result before linking other results"
        ) from invalid_id

    for result in results:
        for reference, item in zip(reference_result["items"], result["items"]):
            item["reference_id"] = reference["id"]
            item["reference_type"] = "annotation"


def create_results_collection(name: Optional[str], item_type: str, values: List[any]) -> dict:
    """
    Create a collection of results for posting to EMPAIA App API.

    :param name: Name of the result collection
    :param item_type: Type of result
    :param values: List of results
    """
    collection = {"item_type": item_type, "items": []}

    for value in values:
        item = {
            "value": value,
        }
        if name is not None:
            item["name"] = name
        if item_type != "class":
            item["type"] = item_type
        collection["items"].append(item)

    return collection


def create_result_scalar(name: str, item_type: str, value: Any) -> dict:
    """
    Create a scalar result for posting to EMPAIA App API.

    :param name: Name of the result
    :param item_type: Type of result
    :param value: Value of the result
    """
    result = {"name": name, "type": item_type, "value": value}
    return result
