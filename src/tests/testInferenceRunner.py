import os
import unittest
from inference import InferenceRunner
from nn import UNet

class TestInferenceRunner(unittest.TestCase):
    def setUp(self):
        self.model_path = "/app/model/hacking_kidney_16934_best_metric.model-384e1332.pth"
        self.sample_image_file = "data/hubmap-kidney-segmentation/test/26dc41664.tiff"

    def test_that_model_loads(self):
        inference = InferenceRunner(self.model_path)
        self.assertIsInstance(inference.model, UNet)

    
