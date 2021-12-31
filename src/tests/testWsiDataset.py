import unittest

import numpy as np
from app.data_classes import TileGetter
from app.tile_loader import TileLoader
from numpy import ndarray
from PIL import Image

from tests.mock_api import MockAPI
from util.combine_tiles import combine_tiles


class TestWSIDataset(unittest.TestCase):
    def setUp(self):
        self.tile_size = (1024, 1024)
        fake_image_data = np.full((2000, 3000, 3), 255)
        api = MockAPI(fake_image_data)

        slide = api.get_input("slide")
        self.roi = api.get_input("region_of_interest")
        self.roi_origin = self.roi["upper_left"]
        self.roi_width = self.roi["width"]
        self.roi_height = self.roi["height"]

        tile_request = lambda x: api.get_wsi_tile(slide, x)

        self.tile_loader = TileLoader(tile_request, self.roi)
        self.roi_image: ndarray = np.asarray(tile_request(self.roi))

    def test_that_tile_loader_provides_a_tile(self):
        first_tile = self.tile_loader[0]["image"]
        self.assertIsInstance(first_tile, ndarray)
        self.assertEqual(first_tile.shape, (*self.tile_size, 3))

    def test_that_combine_tiles_combines_tiles(self):
        tiles = []
        for tile in self.tile_loader:
            tile["image"] = tile["image"][:, :, 0]
            tiles.append(tile)

        combined_tiles = combine_tiles(tiles, self.roi_origin, self.roi_width, self.roi_height)

        self.assertEqual(len(tiles), 6)
        self.assertEqual(combined_tiles.shape, (self.roi_height, self.roi_width))
        self.assertTrue(np.equal(combined_tiles, self.roi_image))
