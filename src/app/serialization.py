from typing import List

from app.entity_extractor import ExtractedEntities


def result_to_collection(result: ExtractedEntities, wsi_id: str, rectangle_id: str):
    polygons = []
    for i, contour in enumerate(result.contours):
        p = {
            "name": f"Glomerulus {i+1}/{result.count()}",
            "description": "Glomerulus with confidence {:.2f}".format(result.confidences[i]),
            "type": "polygon",
            "reference_id": wsi_id,
            "reference_type": "wsi",
            "coordinates": contour,  # Always use WSI base level coordinates
            "npp_created": 500,
            "npp_viewing": [499, 256000],
        }
        polygons.append(p)

    collection = {
        "item_type": "polygon",
        "items": polygons,
        "reference_id": rectangle_id,
        "reference_type": "annotation",
    }

    return collection
