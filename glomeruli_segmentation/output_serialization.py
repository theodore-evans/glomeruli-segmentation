from typing import List, Tuple
from glomeruli_segmentation.data_classes import Rectangle, Wsi

from glomeruli_segmentation.entity_extractor import ExtractedEntities


def result_to_collection(contours: List[Tuple[int,int]], slide: Wsi, roi: Rectangle):
    
    collection = {
        "item_type": "polygon",
        "items": [],
        "reference_id": roi["id"],
        "reference_type": "annotation",
    }
    
    npp = Wsi['pixel_size_nm']
    num_levels = len(Wsi['levels'])
    num_contours = contours.count()
    
    for i, contour in enumerate(contours):
        polygon = {
            "name": f"Glomerulus {i+1}/{num_contours}",
            "description": "Glomerulus",
            "type": "polygon",
            "reference_id": Wsi['id'],
            "reference_type": "wsi",
            "coordinates": contour,  # Always use WSI base level coordinates
            "npp_created": npp,
            "npp_viewing": [npp, npp * 2 ** num_levels],
        }
        collection["items"].append(polygon)

    return collection
