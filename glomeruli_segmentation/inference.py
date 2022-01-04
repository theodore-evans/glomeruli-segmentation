from typing import Iterable, Tuple

import cv2
import numpy as np
import torch
import torch.cuda as cuda
import torch.nn as nn
from torch import Tensor

from glomeruli_segmentation.data_classes import Tile
from glomeruli_segmentation.model.unet import UNet
from glomeruli_segmentation.util.combine_masks import combine_masks


def _ndarray_to_torch_tensor(array: np.ndarray):
    tensor_view = torch.from_numpy(np.atleast_3d(array))
    torch_tensor = tensor_view.permute(2, 0, 1)
    return torch_tensor


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

    def forward(self, input):
        return input[:, self.channel : self.channel + 1, :, :]
    
def load_unet(model_path: str, map_location: str = "cpu"):
    model_data = torch.load(model_path, map_location=map_location)
    unet = UNet(**model_data["kwargs"])
    unet.load_state_dict(model_data["state_dict"])
    return unet


def run_inference(tiles: Iterable[Tile], model: nn.Module, batch_size: int = 1) -> Tile:
    device = torch.device("cuda" if cuda.is_available() else "cpu")
    model = model.to(device).eval()
    torch.set_grad_enabled(False)

    mask_tiles = []

    while True:
        try:
            tensors = []
            rects = []
            for _ in range(batch_size):
                tile: Tile = next(tiles)
                tensors.append(_ndarray_to_torch_tensor(tile.image))
                rects.append(tile.rect)
        except StopIteration:
            break
        finally:
            if len(tensors) == 0:
                break
            input_batch = torch.stack(tensors, dim=0).to(device)
            output_batch: Tensor = model(input_batch)
            for output, rect in zip(output_batch, rects):
                mask = _torch_tensor_to_ndarray(output).squeeze()
                resized_mask = _resize_image(mask, tile.rect.shape)
                mask_tiles.append(Tile(image=resized_mask, rect=rect))

    return combine_masks(mask_tiles)
