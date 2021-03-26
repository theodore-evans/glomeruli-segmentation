from dataclasses import dataclass
from typing import List


@dataclass
class Polygon:
    id: int
    name: str
    reference_id: str  # WSI ID
    reference_type: str  # NO IDEA
    coordinates: List[List[int]]
    type: str = "Polygon"


@dataclass
class Collection:
    description: str
    reference: str
    items: List[Polygon]
    optional: bool = False
    type: str = "collection"


class Output:
    polygons: List[Polygon]
