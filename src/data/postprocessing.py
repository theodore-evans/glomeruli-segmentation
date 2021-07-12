import enum
from typing import List

import numpy as np
from app.data_types import Tile


def combine_tiles(
    tiles: List[Tile],
    upper_left: List[int],
    original_height: int,
    original_width: int,
    ignore_border=0,
):
    """
    Stitch tiles to an array of size original_height x original_width
    """
    combined_image = np.zeros((original_height, original_width), dtype=np.float32)  # type: ignore
    n = np.zeros((original_height, original_width), dtype=np.uint8)  # type: ignore

    for tile in tiles:
        image = tile["image"]
        x = tile["x"] - upper_left[0]
        y = tile["y"] - upper_left[1]

        w = image.shape[1]
        h = image.shape[0]
        xx = x + w
        yy = y + h
        l = 0
        t = 0

        if ignore_border > 0:
            if x > 0:
                x += ignore_border
                l = ignore_border
            if xx < original_width:
                xx -= ignore_border
            if y > 0:
                y += ignore_border
                t = ignore_border
            if yy < original_height:
                yy -= ignore_border
            w = xx - x + l
            h = yy - y + t

        combined_image[y:yy, x:xx] += image[t:h, l:w]
        n[y:yy, x:xx] += 1
    assert np.min(n) > 0

    combined_image /= n
    return combined_image.astype(np.uint8)


def unbatch_predictions(pred_masks: np.ndarray, coords: dict) -> List[Tile]:
    if pred_masks.ndim > 2:
        pred_masks = pred_masks.transpose(2, 0, 1)
        predicted_tiles = [None] * pred_masks.shape[0]
        for idx, pred in enumerate(pred_masks):
            predicted_tiles[idx] = Tile(
                image=pred, x=coords[idx]["x"], y=coords[idx]["y"]
            )
    else:
        predicted_tiles = [
            Tile(image=pred_masks, x=coords[0]["x"], y=coords[0]["y"]
        )]
    return predicted_tiles
