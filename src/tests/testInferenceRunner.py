import os
import unittest
from inference import InferenceRunner
from nn import UNet

class TestInferenceRunner(unittest.TestCase):
    def setUp(self):
        self.model_path = os.path.join("/app/src/model/hacking_kidney_16934_best_metric.model-384e1332.pth")

    def testModelLoads(self):
        inference = InferenceRunner(self.model_path)
        self.assertIsInstance(inference.model, UNet)
