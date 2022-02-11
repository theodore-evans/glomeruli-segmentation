from subprocess import call
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.cuda as cuda
import torch.nn as nn
from koila import lazy
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


# TODO: add lazy tensor loading, potentially using koila (https://github.com/rentruewang/koila)
def run_inference(
    tiles: Iterable[Tile], model: nn.Module, batch_size: int = 1, transform: Optional[Compose] = None
) -> Tile:

    # device = torch.device("cuda" if cuda.is_available() else "cpu")
    device = torch.device("cpu")

    print(f"Running inference on {device}")
    if device == torch.device("cuda"):
        torch.cuda.empty_cache()
        call("nvidia-smi")

    model = model.to(device).eval()

    mask_tiles = []
    while True:
        try:
            tensors = []
            rects = []
            for _ in range(batch_size):
                tile: Tile = next(tiles)
                tensors.append(_ndarray_to_torch_tensor(tile.image, transform))
                rects.append(tile.rect)
        except StopIteration:
            break
        finally:
            torch.set_grad_enabled(False)
            if len(tensors) > 0:
                try:
                    input_batch = lazy(torch.stack(tensors, dim=0).to(device))
                    output_batch: Tensor = model(input_batch)
                except RuntimeError as e:
                    print(f"Runtime Error: {e}, retrying on CPU")
                    input_batch = lazy(torch.stack(tensors, dim=0).to("cpu"))
                    output_batch: Tensor = model.to("cpu")(input_batch)
                for output, rect in zip(output_batch, rects):
                    mask = _torch_tensor_to_ndarray(output).squeeze()
                    resized_mask = _resize_image(mask, tile.rect.shape)
                    mask_tiles.append(Tile(image=resized_mask, rect=rect))

    print(f"Got {len(mask_tiles)} tiles")
    return combine_masks(mask_tiles)
