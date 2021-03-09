from data import KidneyTestDataset
from typing import Optional
import numpy as np
import cv2
from PIL import Image

from data.dataset import Dataset

class WSIDataLoader(KidneyTestDataset):
    def __init__(self, 
                 wsi_tile: Image.Image, 
                 window_size: int = 1024,
                 image_stride: Optional[int] = None,
                 scale: Optional[float] = None
                 ) -> None:
        
        self.scale = scale
        self.window_size = window_size
        self.image_stride = window_size if image_stride is None else image_stride
        self.image = np.asarray_chkfinite(wsi_tile).squeeze()
        
        if self.image.shape[0] == 3:
            self.image = np.transpose(self.image, (1, 2, 0))
        assert self.image.shape[2] == 3 #TODO: error handling here
        
        self.original_width = self.image.shape[1]
        self.original_height = self.image.shape[0]
        
        self.cols = self.width - self.window_size
        self.rows = self.height - self.window_size
        
        self.offset_x = list(range(0, self.width - self.window_size + self.image_stride, self.image_stride))
        self.offset_y = list(range(0, self.height - self.window_size + self.image_stride, self.image_stride))
        
        self.offset_x[-1] = self.shift_offset_into_image(self.offset_x[-1], self.width)
        self.offset_y[-1] = self.shift_offset_into_image(self.offset_y[-1], self.height)
        
        if scale:
            self.image = cv2.resize(self.image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]

    def read(self, x, y, width, height) -> np.ndarray:
        return self.image[y:y + height, x:x + width]
    