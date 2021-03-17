import os
import unittest
import numpy as np
from app.inference_runner import InferenceRunner
from nn import UNet

class TestInferenceRunner(unittest.TestCase):
    def setUp(self):
        self.model_path = "/model/hacking_kidney_16934_best_metric.model-384e1332.pth"
        self.sample_image_file = "data/hubmap-kidney-segmentation/test/26dc41664.tiff"
        self.inference = InferenceRunner(self.model_path)
        self.test_input = np.zeros((1024,1024,3))

    def test_that_model_loads(self):
        self.assertIsInstance(self.inference.model, UNet)

    def test_that_model_accepts_entry_from_wsi_dataset_as_input(self):
        output = self.inference.run_inference_on_image(self.test_input).detach().numpy()
        self.assertEqual(output.shape, self.test_input.shape)
        
