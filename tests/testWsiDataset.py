from app.mock_api import MockAPI
import unittest
import tifffile
from numpy import ndarray

from PIL import Image

from data.wsi_tile_fetcher import WSITileFetcher
from data.postprocessing import combine_tiles

from app.data_types import Rectangle

class TestWSIDataset(unittest.TestCase):
    def setUp(self):
        self.image_size = (2048, 2048)
        sample_image_file = "/data/hubmap-kidney-segmentation/test/26dc41664.tiff"
        self.mock_api = MockAPI(sample_image_file)
        self.wsi_tile_fetcher = WSITileFetcher(self.mock_api.mock_tile_request, self.image_size)

    def test_that_tile_fetcher_provides_a_tile_on_getitem(self):
        first_tile = self.wsi_tile_fetcher[0]["image"]
        self.assertIsInstance(first_tile, ndarray)
        self.assertEqual(first_tile.shape, (1024, 1024, 3))

    def test_that_combine_tiles_combines_tiles(self):
        tiles = []
        for tile in self.wsi_tile_fetcher:
            tile["image"] = tile["image"][:,:,0]
            tiles.append(tile)
            
        combined_tiles = combine_tiles(tiles, *self.image_size)
        self.assertEqual(combined_tiles.shape, self.image_size)
        self.assertEqual(len(tiles), 4)
