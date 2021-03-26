# %%
from app.entity_extractor import EntityExtractor
import os
import unittest
import numpy as np
from app.inference_runner import InferenceRunner
from data.wsi_tile_fetcher import WSITileFetcher
from app.mock_api import MockAPI
from nn import UNet
import cv2
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize

from app.api import API
from app.data_types import Rectangle

TORCH_VISION_MEAN = np.asarray([0.485, 0.456, 0.406])
TORCH_VISION_STD = np.asarray([0.229, 0.224, 0.225])
test_transform = Compose([ToTensor(), Normalize(
    mean=TORCH_VISION_MEAN, std=TORCH_VISION_STD)])

model_path = "/model/hacking_kidney_16934_best_metric.model-384e1332.pth"
sample_image_file = "/data/hubmap-kidney-segmentation/train/54f2eec69.tiff"
inference = InferenceRunner(model_path, data_transform=test_transform)
# %%
# api = API()

# my_rectangle = api.get_input("my_rectangle")
# kidney_wsi = api.get_input("kidney_wsi")

mock_api = MockAPI(sample_image_file)
# %%
upper_left = [15000, 8000]
size_to_process = (3000, 3000)

my_rectangle: Rectangle = {"upper_left": upper_left,
                           "width": size_to_process[0], "height": size_to_process[1], "level": 0}
tile_fetcher = WSITileFetcher(mock_api.mock_tile_request, my_rectangle)
# %%
input_image = np.transpose(np.uint8(
    mock_api.image_data[:, upper_left[0]:upper_left[0]+size_to_process[0], upper_left[1]:upper_left[1] + size_to_process[1]]), (1, 2, 0))
output_mask = np.zeros_like(input_image)
output_mask[:, :, 1] = inference(tile_fetcher)


# TODO get rid of 3 channels for output_mask
entity_extractor = EntityExtractor(upper_left=upper_left)
contours = entity_extractor.extract_contours(output_mask[:,:,1])
count = entity_extractor.count_entities(contours)


# %%
Image.fromarray(output_mask)
# %%

Image.fromarray(input_image)

# %%
alpha = 0.5
superimposed = cv2.addWeighted(input_image, alpha, output_mask, (1 - alpha), 0)
Image.fromarray(superimposed)

# %%
Image.fromarray(tile_fetcher[1]["image"])
# %%
