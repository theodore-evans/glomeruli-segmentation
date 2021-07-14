import asyncio
from typing import NoReturn, Optional, Type

import cv2
import numpy as np
from numpy.core.fromnumeric import ndim, shape
from numpy.core.shape_base import stack
import torch
from torch._C import device, dtype
import torch.cuda as cuda
import torch.nn as nn
from app.data_types import Tile
from data.preprocessing import batch_tiles
from data.postprocessing import combine_tiles, unbatch_predictions
from data.wsi_tile_fetcher import WSITileFetcher
from nn import load_model
from nn.segnet import SegNet
from numpy import ndarray
from torch import Tensor
from torchvision.transforms import Compose


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
        self.device = torch.device("cuda" if cuda.is_available() else "cpu")

        print(f"Using {model_path} on {self.device.type.upper()}")

        self.model: nn.Module = self.load_model(model_path)
        self.transform: Optional[Compose] = data_transform

    def load_model(self, model_path: str) -> SegNet:
        """
        Load a pretrained SegNet model from model_path (.pth extension)
        """
        torch.set_grad_enabled(False)

        model = load_model(model_path).to(self.device)
        model = model.eval()

        return model

    def run_inference_on_image(self, image: ndarray) -> ndarray:
        """
        Run model inference on a single image from a dataset
        """
        if self.transform is not None:
            image_as_tensor = (
                self.transform(image)
                if image.ndim < 4
                else torch.stack(
                    list(map(self.transform, image))
                )
            ) 
        else:
            image_as_tensor = Tensor(image)

        tile_height, tile_width = image_as_tensor.shape[-2:]
        

        model_input = (
            image_as_tensor.unsqueeze(0).to(self.device) 
            if image_as_tensor.ndim < 4 
            else image_as_tensor.to(self.device)
        )

        if model_input.dtype != torch.float32:
            model_input = model_input.float()
        model_output = self.model(model_input)
        
        #a.transpose(0, 3).transpose(0,2).transpose(0,1).shape
        pixelwise_probabilities = nn.functional.softmax(model_output).cpu()[:, 1, :, :].numpy()
        pixelwise_probabilities = pixelwise_probabilities.transpose(1, 2, 0)
        resized_probabilities = cv2.resize(
            pixelwise_probabilities,
            (tile_height, tile_width),
            interpolation=cv2.INTER_LINEAR,
        )
        prediction_mask = (resized_probabilities * 255).astype(dtype=np.uint8)  # type: ignore
        return prediction_mask

    def run_inference_on_tile(self, tile: Tile) -> Tile:
        # print(f"\nRunning inference on tile with x: {tile['x']}, y: {tile['y']}")
        predicted_mask = self.run_inference_on_image(tile["image"])
        predicted_tile: Tile = {"image": predicted_mask, "x": tile["x"], "y": tile["y"]}
        # print("\nDone")
        return predicted_tile

    def run_inference_on_dataset(self, tile_fetcher: Type[WSITileFetcher]) -> ndarray:
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

    def run_batched_inference_on_dataset(self, tile_fetcher: Type[WSITileFetcher]) -> ndarray:
        img_batch, coords = batch_tiles(tile_fetcher)
        pred_masks = self.run_inference_on_image(img_batch)
        predicted_tiles = unbatch_predictions(pred_masks, coords)
        return combine_tiles(
            predicted_tiles,
            tile_fetcher.upper_left,
            tile_fetcher.height,
            tile_fetcher.width,
        )

    def __call__(self, input_dataset) -> ndarray:
        return self.run_inference_on_dataset(input_dataset)

    