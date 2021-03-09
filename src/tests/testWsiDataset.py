import os
import unittest
import tifffile

import cv2
from PIL import Image

from data.wsi_data_loader import WSIDataLoader

class TestWSIDataset(unittest.TestCase):
    def setUp(self):
        sample_image_file = "/data/hubmap-kidney-segmentation/test/26dc41664.tiff"
        self.sample_image = Image.fromarray(tifffile.imread(sample_image_file))
        self.dataset = WSIDataLoader(self.sample_image)

    def test_that_dataset_is_created_from_image(self):
        self.assertTrue(self.dataset.width != 0)
        
    def test_that_dataset_entries_are_images_of_correct_size(self):
        pass
