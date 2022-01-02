# from logging import Logger
# from typing import Iterator, Optional

from typing import Iterable

import cv2
import numpy as np
import torch
import torch.cuda as cuda
import torch.nn as nn
from app.data_classes import Tile
from torch import Tensor

from util.combine_masks import combine_masks

# from torchvision.transforms import Compose


def _ndarray_to_torch_tensor(array: np.ndarray):
    tensor_view = torch.from_numpy(np.atleast_3d(array))
    torch_tensor = tensor_view.permute(2, 0, 1).unsqueeze(0)
    return torch_tensor


def _torch_tensor_to_ndarray(torch_tensor: Tensor):
    numpy_format = torch_tensor.squeeze(0).permute(1, 2, 0)
    array = np.array(numpy_format)
    return array


def _scale_image_to_fit_tile(tile: Tile, interpolation: int = cv2.INTER_LINEAR) -> Tile:
    resized = cv2.resize(
        tile.image,
        tile.rect.shape,
        interpolation=interpolation,
    )
    return Tile(image=resized, rect=tile.rect)


def run_inference(tiles: Iterable[Tile], model: nn.Module, batch_size: int=1) -> Tile:
    device = torch.device("cuda" if cuda.is_available() else "cpu")
    model = model.to(device).eval()

    mask_tiles = []
    for tile in tiles:
        tensor = _ndarray_to_torch_tensor(tile.image)
        output = model(tensor)
        mask = _torch_tensor_to_ndarray(output).squeeze()
        mask_tile = _scale_image_to_fit_tile(Tile(image=mask, rect=tile.rect))
        mask_tiles.append(mask_tile)

    return combine_masks(mask_tiles)


# class InferenceRunner:
#     def __init__(
#         self,
#         model_path: str,
#         data_transform: Optional[Compose] = raw_test_transform,
#         logger: Logger = get_logger(),
#     ) -> None:
#         """
#         Description:
#             An inference runner for the pre-trained segmentation model (.pth extension) \
#         located at <model_path>. Runs segmentation using the provided model, returning an combined numpy.ndarray \
#         of the results.

#         Arguments:
#             model_path: Absolute filepath to pretrained segmentation model (type nn.SegNet, file extension .pth)
#             data_transform: Optional transform of type torchvision.transforms.Compose to apply before inference

#         """

#         self.device = torch.device("cuda" if cuda.is_available() else "cpu")

#         self.logger.info(f"Using {model_path} on {self.device.type.upper()}")

#         self.model: nn.Module = self.load_model(model_path)
#         self.transform: Optional[Compose] = data_transform
#         self.logger = logger

#

#     def run_inference_on_image(self, image: ndarray) -> ndarray:
#         """
#         Run model inference on a single image from a dataset
#         """
#         image_as_tensor = self.transform(image) if self.transform else Tensor(image)

#         model_input = image_as_tensor.unsqueeze(0).to(self.device)

#         model_output = self.model(model_input)

#         pixelwise_probabilities = nn.functional.softmax(model_output).cpu()[0, 1, :, :].numpy()
#         resized_probabilities = cv2.resize(
#             pixelwise_probabilities,
#             image_as_tensor.shape,
#             interpolation=cv2.INTER_LINEAR,
#         )
#         prediction_mask = (resized_probabilities * 255).astype(dtype=np.uint8)  # type: ignore
#         return prediction_mask

#     def run_inference_on_tile(self, tile: Tile) -> Tile:
#         self.logger.info(f"\nRunning inference on tile with x: {tile['x']}, y: {tile['y']}")
#         predicted_mask = self.run_inference_on_image(tile["image"])
#         predicted_tile: Tile = {"image": predicted_mask, "rect": tile["rect"]}
#         self.logger.info("\nDone")
#         return predicted_tile

#     def run_inference_on_dataset(self, tile_loader: Iterator[Tile]) -> ndarray:
#         """
#         Fetch and run model inference on WSI tiles and combine results into
#         a single ndarray of the same width and height as the original WSI tile
#         """

#         # TODO: add batching here
#         predicted_tiles = []
#         for tile in tile_loader:
#             # tile["image"] = np.transpose(tile["image"], (2, 0, 1))
#             predicted_tiles.append(self.run_inference_on_tile(tile))

#         return combine_tiles(predicted_tiles)["image"]

#     def __call__(self, input_dataset) -> ndarray:
#         return self.run_inference_on_dataset(input_dataset)
