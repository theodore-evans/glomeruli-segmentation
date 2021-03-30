from dataclasses import dataclass
from typing import List


@dataclass
class Polygon:
    id: str
    name: str
    reference_id: str  # WSI ID
    reference_type: str  # NO IDEA
    coordinates: List[List[int]]
    type: str = "Polygon"


@dataclass
class Collection:
    name: str
    description: str
    reference: str
    items: List[Polygon]
    optional: bool = False
    type: str = "collection"


class Output:
    polygons: List[Polygon]


def contours_to_collection(contours: List, wsi_id: str, rectangle_id: str):
    polygons = []
    for i, contour in enumerate(contours):
        p = Polygon(id=f'contour_{i}', name=f'Contour {i}',
                    reference_id=wsi_id, reference_type="No idea", coordinates=contour, type="Polygon")
        polygons.append(p)

    c = Collection(description="Glomeruli Annotations",
                   reference="inputs.kidney_wsi",  # or rectangle_id????
                   items=polygons,
                   optional=True,
                   type="collection",
                   name="Glomeruli Polygons")

    return c
