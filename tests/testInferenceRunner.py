import os
import unittest
import numpy as np
from app.inference_runner import InferenceRunner
from data.wsi_tile_fetcher import WSITileFetcher
from app.mock_api import MockAPI
from nn import UNet

class TestInferenceRunner(unittest.TestCase):
    def setUp(self):
        self.model_path = "/model/hacking_kidney_16934_best_metric.model-384e1332.pth"
        self.sample_image_file = "/data/hubmap-kidney-segmentation/test/26dc41664.tiff"
        self.inference = InferenceRunner(self.model_path)
        self.test_input = np.zeros((3,1024,1024))
        mock_api = MockAPI(self.sample_image_file)
        self.tile_fetcher = WSITileFetcher(mock_api.mock_tile_request, (2048,2048))

    def test_that_model_loads(self):
        self.assertIsInstance(self.inference.model, UNet)

    def test_that_inference_runs_on_empty_tensor(self):
        output = self.inference.run_inference_on_image(self.test_input)
        self.assertEqual(output.shape, self.test_input.shape[1:])
        
    def test_that_inference_runs_on_single(self):
        input_image = np.transpose(self.tile_fetcher[0]["image"], (2,0,1))
        output = self.inference.run_inference_on_image(input_image)
        self.assertEqual(output.shape, (1024,1024))
    
    def test_that_inference_runs_on_tile_fetcher(self):
        output = self.inference(self.tile_fetcher)
        print(output.shape)
        
