from logging import Logger
from typing import Iterator, Optional

import cv2
import numpy as np
import torch
import torch.cuda as cuda
import torch.nn as nn
from app.data_classes import Tile
from app.logging_tools import get_logger
from app.tile_loader import TileLoader
from data.preprocessing import raw_test_transform
from nn import load_model
from nn.segnet import SegNet
from numpy import ndarray
from torch import Tensor
from torchvision.transforms import Compose

from util.combine_tiles import combine_tiles


class InferenceRunner:
    def __init__(
        self,
        model_path: str,
        data_transform: Optional[Compose] = raw_test_transform,
        logger: Logger = get_logger(),
    ) -> None:
        """
        Description:
            An inference runner for the pre-trained segmentation model (.pth extension) \
        located at <model_path>. Runs segmentation using the provided model, returning an combined numpy.ndarray \
        of the results.

        Arguments:
            model_path: Absolute filepath to pretrained segmentation model (type nn.SegNet, file extension .pth)
            data_transform: Optional transform of type torchvision.transforms.Compose to apply before inference

        """

        self.device = torch.device("cuda" if cuda.is_available() else "cpu")

        self.logger.info(f"Using {model_path} on {self.device.type.upper()}")

        self.model: nn.Module = self.load_model(model_path)
        self.transform: Optional[Compose] = data_transform
        self.logger = logger

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
        image_as_tensor = self.transform(image) if self.transform else Tensor(image)

        model_input = image_as_tensor.unsqueeze(0).to(self.device)

        model_output = self.model(model_input)

        pixelwise_probabilities = nn.functional.softmax(model_output).cpu()[0, 1, :, :].numpy()
        resized_probabilities = cv2.resize(
            pixelwise_probabilities,
            image_as_tensor.shape,
            interpolation=cv2.INTER_LINEAR,
        )
        prediction_mask = (resized_probabilities * 255).astype(dtype=np.uint8)  # type: ignore
        return prediction_mask

    def run_inference_on_tile(self, tile: Tile) -> Tile:
        self.logger.info(f"\nRunning inference on tile with x: {tile['x']}, y: {tile['y']}")
        predicted_mask = self.run_inference_on_image(tile["image"])
        predicted_tile: Tile = {"image": predicted_mask, "rect": tile["rect"]}
        self.logger.info("\nDone")
        return predicted_tile

    def run_inference_on_dataset(self, tile_loader: Iterator[Tile]) -> ndarray:
        """
        Fetch and run model inference on WSI tiles and combine results into
        a single ndarray of the same width and height as the original WSI tile
        """

        # TODO: add batching here
        predicted_tiles = []
        for tile in tile_loader:
            # tile["image"] = np.transpose(tile["image"], (2, 0, 1))
            predicted_tiles.append(self.run_inference_on_tile(tile))

        return combine_tiles(predicted_tiles)["image"]

    def __call__(self, input_dataset) -> ndarray:
        return self.run_inference_on_dataset(input_dataset)
