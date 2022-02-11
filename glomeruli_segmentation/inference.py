import logging
from subprocess import call
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.cuda as cuda
import torch.nn as nn
from torch import Tensor
from torchvision.transforms.transforms import Compose

from glomeruli_segmentation.data_classes import Tile
from glomeruli_segmentation.logging_tools import get_logger
from glomeruli_segmentation.model.unet import UNet
from glomeruli_segmentation.util.combine_masks import combine_masks


def _ndarray_to_torch_tensor(array: np.ndarray, transform: Optional[Compose] = None):
    tensor_view = torch.from_numpy(np.atleast_3d(array))
    torch_tensor = tensor_view.permute(2, 0, 1)
    return transform(torch_tensor) if transform else torch_tensor


def _torch_tensor_to_ndarray(torch_tensor: Tensor):
    numpy_format = torch_tensor.permute(1, 2, 0).detach().to("cpu")
    array = np.array(numpy_format)
    return array


def _resize_image(
    image: np.ndarray, target_shape: Tuple[int, int], interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    return cv2.resize(
        image,
        target_shape,
        interpolation=interpolation,
    )


class SingleChannelPassthrough(nn.Module):
    def __init__(self, channel: int = 0):
        super().__init__()
        self.channel = channel

    def forward(self, inputs):
        return inputs[:, self.channel : self.channel + 1, :, :]


def load_unet(model_path: str, map_location: str = "cpu"):
    model_data = torch.load(model_path, map_location=map_location)
    unet = UNet(**model_data["kwargs"])
    unet.load_state_dict(model_data["state_dict"])
    return unet


def run_inference_on_tiles(tiles: Iterable[Tile], model: nn.Module, transform: Optional[Compose] = None) -> Tile:

    device = torch.device("cpu")

    logger = get_logger("inference", log_level=logging.INFO)
    logger.info(f"Running inference on {str(device).upper()}")

    if device == torch.device("cuda"):
        torch.cuda.empty_cache()
        call("nvidia-smi")

    model = model.to(device).eval()

    # TODO: add GPU support and lazy tensor loading
    mask_tiles = []
    for tile in tiles:
        array = np.array(tile.image, dtype=np.float32) / 255
        inputs = _ndarray_to_torch_tensor(array)[None, :, :, :].to(device)
        inputs = transform(inputs) if transform else inputs
        output = model(inputs)
        output = _torch_tensor_to_ndarray(torch.squeeze(output, 0))
        resized = _resize_image(output, tile.rect.shape)
        mask_tiles.append(Tile(image=resized, rect=tile.rect))

    logger.info(f"Combining {len(mask_tiles)} tiles")
    return combine_masks(mask_tiles)
