from typing import Callable, Iterable, Optional, Tuple, Union

from glomeruli_segmentation.data_classes import Rectangle, Tile, Vector2

WindowShape = Union[Vector2, Tuple[int, int]]


def get_tile_loader(
    get_tile: Callable[[Rectangle], Tile],
    region: Rectangle,
    window: WindowShape,
    stride: Optional[WindowShape] = None,
) -> Iterable[Tile]:

    window = Vector2(*window)
    stride = Vector2(*stride) if stride else window

    start_x, start_y = (region.upper_left.x, region.upper_left.y)
    effective_shape = Vector2(region.width - window.x + stride.x, region.height - window.y + stride.y)

    whole_columns = effective_shape.x // stride.x
    columns_remainder = effective_shape.x % stride.x

    whole_rows = effective_shape.y // stride.y
    rows_remainder = effective_shape.y % stride.y

    corners = []

    for j in range(whole_rows + (rows_remainder > 0)):
        for i in range(whole_columns + (columns_remainder > 0)):
            x, y = (start_x + i * stride.x, start_y + j * stride.y)

            if i - whole_columns >= 0:
                x = region.width - window.x
            if j - whole_rows >= 0:
                y = region.height - window.y

            corners.append((x, y))

    rectangles = [
        Rectangle(upper_left=corner, width=window.x, height=window.y, level=region.level)
        for corner in corners
    ]

    return (get_tile(rect) for rect in rectangles)
