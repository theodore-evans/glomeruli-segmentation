import numpy as np
import cv2
from PIL import Image

from data.dataset import Dataset

class WSIDataset(Dataset):
    def __init__(self, wsi_tile: Image, scale=None):
        self.scale = scale
        self.image = np.asarray(wsi_tile).squeeze()
        if self.image.shape[0] == 3:
            self.image = np.transpose(self.image, (1, 2, 0))
        assert self.image.shape[2] == 3
        self.original_width = self.image.shape[1]
        self.original_height = self.image.shape[0]
        if scale:
            self.image = cv2.resize(self.image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    @property
    def height(self):
        return self.image.shape[0]

    @property
    def width(self):
        return self.image.shape[1]

    def read(self, x, y, width, height):
        return self.image[y:y + height, x:x + width]