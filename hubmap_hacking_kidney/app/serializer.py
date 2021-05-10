from typing import List


def contours_to_collection(contours: List, wsi_id: str, rectangle_id: str):
    polygons = []
    for i, contour in enumerate(contours):
        p = {
            "name": f"Glomerulus {i}",
            "type": "polygon",
            "reference_id": wsi_id,  # each point annotation references my_wsi
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
