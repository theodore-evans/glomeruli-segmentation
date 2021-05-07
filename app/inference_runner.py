import asyncio
from typing import NoReturn, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from numpy import ndarray
from torch import Tensor
from torch.cuda import is_available
from torchvision.transforms import Compose

from app.data_types import Tile
from data.postprocessing import combine_tiles
from data.wsi_tile_fetcher import WSITileFetcher
from nn import load_model
from nn.segnet import SegNet


class InferenceRunner:
    def __init__(self, model_path: str, data_transform: Optional[Compose] = None) -> None:
        """
        Description:
            An inference runner for the pre-trained segmentation model (.pth extension) \
        located at <model_path>. When called with a WSITileFetcher, asynchronously fetches \
        WSI tiles and runs segmentation using the provided model, returning an combined numpy.ndarray \
        of the results.

        Arguments:
            model_path: Absolute filepath to pretrained segmentation model (type nn.SegNet, file extension .pth)
            data_transform: Optional transform of type torchvision.transforms.Compose to apply before inference

        """
        self.model: nn.Module = self.load_model(model_path)
        self.transform: Optional[Compose] = data_transform

    def load_model(self, model_path: str) -> SegNet:
        """
        Load a pretrained SegNet model from model_path (.pth extension)
        """
        torch.set_grad_enabled(False)
        model = load_model(model_path)
        if torch.cuda.is_available():
            model = model.cuda()
        return model.eval()

    def run_inference_on_image(self, image: ndarray) -> ndarray:
        """
        Run model inference on a single image from a dataset
        """
        if self.transform is not None:
            image_as_tensor = self.transform(image)
        else:
            image_as_tensor = Tensor(image)

        tile_height = image_as_tensor.shape[1]
        tile_width = image_as_tensor.shape[2]

        if torch.cuda.is_available():
            model_input = image_as_tensor.cuda()

        model_input = model_input.unsqueeze(0)
        model_output = self.model(model_input)

        pixelwise_probabilities = nn.functional.softmax(model_output).cpu()[0, 1, :, :].numpy()
        resized_probabilities = cv2.resize(
            pixelwise_probabilities,
            (tile_height, tile_width),
            interpolation=cv2.INTER_LINEAR,
        )
        prediction_mask = (resized_probabilities * 255).astype(dtype=np.uint8)  # type: ignore
        return prediction_mask

    def run_inference_on_tile(self, tile: Tile) -> Tile:
        print(f"\nRunning inference on tile with x: {tile['x']}, y: {tile['y']}")
        predicted_mask = self.run_inference_on_image(tile["image"])
        predicted_tile: Tile = {"image": predicted_mask, "x": tile["x"], "y": tile["y"]}
        print("\nDone")
        return predicted_tile

    def run_inference_on_dataset(self, tile_fetcher: WSITileFetcher) -> ndarray:
        """
        Fetch and run model inference on WSI tiles and combine results into
        a single ndarray of the same width and height as the original WSI tile
        """

        predicted_tiles = []
        for tile in tile_fetcher:
            # tile["image"] = np.transpose(tile["image"], (2, 0, 1))
            predicted_tiles.append(self.run_inference_on_tile(tile))

        # TODO inject rather than hardcode
        return combine_tiles(
            predicted_tiles,
            tile_fetcher.upper_left,
            tile_fetcher.height,
            tile_fetcher.width,
        )

    def __call__(self, input_dataset) -> ndarray:
        return self.run_inference_on_dataset(input_dataset)
