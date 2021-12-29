from typing import List, Tuple

import numpy as np
from app.data_types import Rectangle, Tile


def combine_tiles(
    tiles: List[Tile],
    region_upper_left: Tuple[int, int],
    original_width: int,
    original_height: int,
    ignore_border=0,
):
    """
    Stitch tiles to an array of size original_height x original_width
    """
    raise NotImplementedError
    # combined_image = np.zeros((original_height, original_width), dtype=np.float32)  # type: ignore
    # n = np.zeros((original_height, original_width), dtype=np.uint8)  # type: ignore
    # print(f"{region_upper_left=}")
    # for tile in tiles:
    #     image: np.ndarray = tile["image"]
        
    #     tile_upper_left = (tile["x"], tile["y"])
        
    #     local_origin = tile_upper_left - region_upper_left

    #     tile_width = image.shape[0]
    #     tile_height = image.shape[1]
        
    #     xx = local_origin[0] + tile_width
    #     yy = local_origin[1] + tile_height
        
    #     l = 0
    #     t = 0

    #     if ignore_border > 0:
    #         if local_origin > 0:
    #             local_origin += ignore_border
    #             l = ignore_border
    #         if xx < original_width:
    #             xx -= ignore_border
    #         if local_y > 0:
    #             local_y += ignore_border
    #             t = ignore_border
    #         if yy < original_height:
    #             yy -= ignore_border
    #         tile_width = xx - local_origin + l
    #         tile_height = yy - local_y + t

    #     print(f"{tile_width=}, {tile_height=}, {tile['x']=}, {tile['y']=}, {local_origin=}, {local_y=}, {xx=}, {yy=}")
        
    #     combined_image[local_origin:xx, local_y:yy] += image[l:tile_width, t:tile_height]
    #     n[local_origin:xx, local_y:yy] += 1
    # assert np.min(n) > 0

    # combined_image /= n
    # return combined_image.astype(np.uint8)
