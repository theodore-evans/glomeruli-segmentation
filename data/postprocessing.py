import numpy as np
from typing import List

from app.data_types import Tile

def combine_tiles(tiles: List[Tile],
                  original_height: int,
                  original_width: int,
                  ignore_border=0):
        """
        Stitch tiles to an array of size original_height x original_width
        """
        combined_image = np.zeros((original_height, original_width), dtype=np.float32) #type: ignore
        n = np.zeros((original_height, original_width), dtype=np.uint8) #type: ignore
        
        for tile in tiles:
            image = tile["image"]
            x = tile["x"]
            y = tile["y"]

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