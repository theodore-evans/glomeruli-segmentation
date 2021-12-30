from typing import Iterable, Optional, Tuple

from app.data_types import Rectangle, Tile, TileGetter


def get_tile_loader(
    get_tile: TileGetter, region: Rectangle, window: Tuple[int, int], stride: Optional[Tuple[int, int]] = None
) -> Iterable[Tile]:

    stride = stride if stride else window

    start_x, start_y = region["upper_left"]
    region_shape = region["width"], region["height"]
    effective_shape = tuple(side - window[dim] + stride[dim] for dim, side in enumerate(region_shape))

    whole_columns = effective_shape[0] // stride[0]
    columns_remainder = effective_shape[0] % stride[0]

    whole_rows = effective_shape[1] // stride[1]
    rows_remainder = effective_shape[1] % stride[1]

    corners = []

    for j in range(whole_rows + (rows_remainder > 0)):
        for i in range(whole_columns + (columns_remainder > 0)):
            x, y = (start_x + i * stride[0], start_y + j * stride[1])

            if i - whole_columns >= 0:
                x = region_shape[0] - window[0]
            if j - whole_rows >= 0:
                y = region_shape[1] - window[1]

            corners.append((x, y))

    rectangles = [
        Rectangle(upper_left=corner, width=window[0], height=window[1], level=region["level"])
        for corner in corners
    ]

    return (get_tile(rect) for rect in rectangles)