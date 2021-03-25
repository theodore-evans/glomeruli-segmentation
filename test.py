#%%
import os
import unittest
import numpy as np
from app.inference_runner import InferenceRunner
from data.wsi_tile_fetcher import WSITileFetcher
from app.mock_api import MockAPI
from nn import UNet
import PIL.Image
from PIL import Image

model_path = "/model/hacking_kidney_16934_best_metric.model-384e1332.pth"
sample_image_file = "/data/hubmap-kidney-segmentation/train/54f2eec69.tiff"
inference = InferenceRunner(model_path)
test_input = np.zeros((3,1024,1024))
upper_left = (14000,6000)
size_to_process = (4096, 4096)
mock_api = MockAPI(sample_image_file, upper_left)
tile_fetcher = WSITileFetcher(mock_api.mock_tile_request, size_to_process)
#%%
output = inference(tile_fetcher)
# %%
Image.fromarray(output)
# %%
Image.fromarray(np.transpose(np.uint8(mock_api.image_data[:, 14000:18096, 6000:10044]), (1,2,0)))

# %%
