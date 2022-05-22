import numpy as np

from glomeruli_segmentation.data_classes import Rect, Tile


def make_tile(rect: Rect):
    width, height = (rect.width, rect.height)
    image = np.arange(width * height, dtype=np.float32)
    image = image.reshape((width, height))
    image *= 255 / (width * height)
    return Tile(image=image, rect=rect)


def make_tile_getter(tile: Tile):
    def tile_getter(rect: Rect):
        x_start = rect.upper_left.x
        x_end = x_start + rect.width
        y_start = rect.upper_left.y
        y_end = y_start + rect.height
        return Tile(image=tile.image.copy()[x_start:x_end, y_start:y_end], rect=rect)

    return tile_getter
