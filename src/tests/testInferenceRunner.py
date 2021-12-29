import os
import unittest

import numpy as np
from app.inference import InferenceRunner
from tests.mock_api import MockAPI
from app.tile_loader import TileLoader
from nn import UNet


class TestInferenceRunner(unittest.TestCase):
    def setUp(self):
        self.model_path = "/model/hacking_kidney_16934_best_metric.model-384e1332.pth"
        self.inference = InferenceRunner(self.model_path)
        self.test_input = np.zeros((3, 1024, 1024))
        mock_api = MockAPI(self.sample_image_file)
        self.tile_fetcher = TileLoader(mock_api.mock_tile_request, (2048, 2048))

    def test_that_model_loads(self):
        self.assertIsInstance(self.inference.model, UNet)

    def test_that_inference_runs_on_empty_tensor(self):
        output = self.inference.run_inference_on_image(self.test_input)
        self.assertEqual(output.shape, self.test_input.shape[1:])

    def test_that_inference_runs_on_single(self):
        input_image = np.transpose(self.tile_fetcher[0]["image"], (2, 0, 1))
        output = self.inference.run_inference_on_image(input_image)
        self.assertEqual(output.shape, (1024, 1024))

    def test_that_inference_runs_on_tile_fetcher(self):
        output = self.inference(self.tile_fetcher)
        print(output.shape)
