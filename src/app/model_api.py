from app.inference_runner import InferenceRunner
from data.tile_fetcher import TileFetcher

import numpy as np

class ModelAPI():
    def __init__(self, inference: InferenceRunner, tile_fetcher: TileFetcher) -> None:
        self.inference_runner = inference
        self.tile_fetcher = tile_fetcher

    def __call__(self, image: np.ndarray, *args, **kwds) -> np.ndarray: 
        self.tile_fetcher.region = image
        return self.inference_runner.run_batched_inference_on_dataset(self.tile_fetcher)