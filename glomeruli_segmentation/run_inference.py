import logging
import time
from subprocess import call
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as fn
from torch import Tensor
from torchvision.transforms.transforms import Compose

from glomeruli_segmentation.data_classes import Tile
from glomeruli_segmentation.logging_tools import get_logger
from glomeruli_segmentation.model.unet import UNet
from glomeruli_segmentation.util.combine_masks import combine_masks


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


def _ndarray_to_torch_tensor(array: np.ndarray, transform: Optional[Compose] = None):
    tensor_view = torch.as_tensor(np.atleast_3d(array))
    torch_tensor = tensor_view.permute(2, 0, 1)
    return transform(torch_tensor) if transform else torch_tensor


def _torch_tensor_to_ndarray(torch_tensor: Tensor):
    torch_tensor = torch.squeeze(torch_tensor, 0)
    numpy_format = torch_tensor.permute(1, 2, 0).detach().cpu()
    array = np.array(numpy_format)
    return array


def run_inference_on_tiles(tiles: Iterable[Tile], model: nn.Module, transform: Optional[Compose] = None) -> Tile:

    if torch.cuda.is_available():
        device = torch.device("cuda")
        call("nvidia-smi")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    logger = get_logger("inference", log_level=logging.INFO)
    logger.info(f"Running inference on {str(device).upper()}")
    start_time = time.time()

    model = model.type(dtype).to(device).eval()

    mask_tiles = []
    iteration_average_time = 0

    for tile in tiles:
        iteration_start_time = time.time()
        array = np.array(tile.image, dtype=np.float32) / 255
        with torch.no_grad():
            inputs = _ndarray_to_torch_tensor(array)[None, :, :, :].type(dtype).to(device)
            inputs = transform(inputs) if transform else inputs
            output = model(inputs)
            resized_output = fn.resize(output, list(tile.rect.shape))
            resized_output = _torch_tensor_to_ndarray(resized_output)
        mask_tiles.append(Tile(image=resized_output, rect=tile.rect))
        iteration_average_time += time.time() - iteration_start_time

    elapsed_time = time.time() - start_time
    number_of_iterations = len(mask_tiles)
    iteration_average_time /= number_of_iterations

    logger.info(f"Inference completed in {round(elapsed_time, 1)}s, tile average {round(iteration_average_time, 1)}s")

    combined = combine_masks(mask_tiles)
    logger.info(f"Combined {len(mask_tiles)} tiles")
    return combined
