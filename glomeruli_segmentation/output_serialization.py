from typing import List, Tuple

from glomeruli_segmentation.data_classes import Rectangle, Wsi


def serialize_result_to_collection(contours: List[Tuple[int, int]], slide: Wsi, roi: Rectangle):

    npp = slide.pixel_size_nm.x
    num_levels = len(slide.levels)
    num_contours = len(contours)

    polygons: List[dict] = []

    for i, contour in enumerate(contours):
        polygon = {
            "name": f"Glomerulus {i+1}/{num_contours}",
            "description": "Glomerulus",
            "type": "polygon",
            "reference_id": slide.id,
            "reference_type": "wsi",
            "coordinates": contour,  # Always use WSI base level coordinates
            "npp_created": npp,
            "npp_viewing": [npp, npp * 2 ** num_levels],
        }
        polygons.append(polygon)

    return {
        "item_type": "polygon",
        "items": polygons,
        "reference_id": roi.id,
        "reference_type": "annotation",
    }
