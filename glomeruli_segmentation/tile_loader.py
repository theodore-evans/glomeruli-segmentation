from typing import Callable, Iterable, Optional, Tuple

from glomeruli_segmentation.data_classes import Rectangle, Tile, Vector2


def get_tile_loader(
    get_tile: Callable[[Rectangle], Tile],
    region: Rectangle,
    window: Tuple[int, int],
    stride: Optional[Tuple[int, int]] = None,
) -> Iterable[Tile]:

    window_width, window_height = window
    stride_x, stride_y = stride if stride else window

    start_x, start_y = region.upper_left
    effective_shape = Vector2((region.width - window_width + stride_x, region.height - window_height + stride_y))

    whole_columns = effective_shape.x // stride_x
    columns_remainder = effective_shape.x % stride_x

    whole_rows = effective_shape.y // stride_y
    rows_remainder = effective_shape.y % stride_y

    corners = []

    for j in range(whole_rows + (rows_remainder > 0)):
        for i in range(whole_columns + (columns_remainder > 0)):
            x, y = (start_x + i * stride_x, start_y + j * stride_y)

            if i - whole_columns >= 0:
                x = start_x + region.width - window_width
            if j - whole_rows >= 0:
                y = start_y + region.height - window_height

            corners.append(Vector2((x, y)))

    rectangles = [
        Rectangle(upper_left=corner, width=window_width, height=window_height, level=region.level) for corner in corners
    ]

    return (get_tile(rect) for rect in rectangles)
