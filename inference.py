from typing import List
import torch
import torch.nn
import cv2
import copy

import numpy as np
from numpy import ndarray
from torchvision.transforms import Compose, ToTensor, Normalize
from torch import Tensor

from nn import load_model
from nn.segnet import SegNet
from data.wsi_data_loader import Tile, WSITileDataset, combine_tiles
from data.data_processing import kaggle_test_transform

class InferenceRunner:
    def __init__(self, model_path: str, data_transform: Compose = kaggle_test_transform()) -> None:
        """
        TODO: write docstring
        """
        self.model = load_model(model_path)
        self.transform = data_transform

    def load_model(self, model_path: str) -> SegNet:
        """
        Load a pretrained SegNet model from model_path (.pth extension)
        """
        torch.set_grad_enabled(False)
        model = load_model(model_path)
        model = model.cuda().eval()
        return model

    def run_inference_on_tile_image(self, tile_image: ndarray) -> ndarray:
        """
        Run model inference on a single image from a dataset
        """
        image_as_tensor = Tensor(tile_image)
        tile_height, tile_width = image_as_tensor.shape[:2]
        
        model_input = self.transform(image_as_tensor).unsqueeze(0).cuda()
        model_output = self.model(model_input)
        
        pixelwise_probabilities = torch.nn.functional.softmax(model_output).cpu()[0, 1, :, :].numpy()
        resized_probabilities = cv2.resize(pixelwise_probabilities, (tile_height, tile_width), interpolation=cv2.INTER_LINEAR)
        prediction_mask = (resized_probabilities * 255).astype(dtype=np.uint8) #type: ignore
        return  prediction_mask

    def run_inference_on_dataset(self, dataset: WSITileDataset) -> ndarray:
        """
        Run model inference on all tiles from dataset and combine results into 
        a single ndarray of the same width and height as the original WSI tile
        """
        predicted_tiles: List[Tile] = []
        for tile in dataset:
            predicted_tile: Tile = copy.deepcopy(tile)
            predicted_mask = self.run_inference_on_tile_image(tile["image"])
            predicted_tile["image"] = predicted_mask
            predicted_tiles.append(predicted_tile)
            
        original_height = dataset.original_height
        original_width = dataset.original_width
        return combine_tiles(predicted_tiles, original_height, original_width)
        
    def __call__(self, inputs) -> ndarray:
        return self.run_inference_on_tile_image(inputs)