import unittest
import tifffile

from PIL import Image

from data.wsi_data_loader import WSITileDataset

class TestWSIDataset(unittest.TestCase):
    def setUp(self):
        self.tile_width = 1024
        sample_image_file = "/data/hubmap-kidney-segmentation/test/26dc41664.tiff"
        self.sample_image = Image.fromarray(tifffile.imread(sample_image_file).squeeze().transpose(1,2,0))
        self.dataset = WSITileDataset(wsi_tile=self.sample_image, window_size=self.tile_width)

    def test_that_dataset_is_created_from_image(self):
        self.assertTrue(self.dataset.width != 0)
        
    def test_that_dataset_entries_are_arrays_of_correct_size(self):
        self.assertEqual(self.dataset[0]["image"].shape[0], self.tile_width)
        
