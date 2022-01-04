import numpy as np
import torch
import torch.nn as nn

from glomeruli_segmentation.data_classes import Rectangle, Tile
from glomeruli_segmentation.inference import load_unet, run_inference, SingleChannelPassthrough
from glomeruli_segmentation.tile_loader import get_tile_loader
from tests.helper_methods import make_tile, make_tile_getter


one_channel_passthrough = SingleChannelPassthrough()

downscaling_model = nn.Sequential(
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1), SingleChannelPassthrough(channel=0)
)


def test_returns_a_mask_for_a_single_tile():
    rect = Rectangle(upper_left=(0, 0), width=128, height=128)
    input_tile = make_tile(rect)
    tile_loader = iter([input_tile])

    mask = run_inference(tile_loader, one_channel_passthrough)
    assert isinstance(mask, Tile)
    assert mask.image.shape == input_tile.image.shape[:2]
    assert mask.rect == rect


def test_returns_a_combined_mask_for_split_tiles():
    rect = Rectangle(upper_left=(0, 0), width=256, height=256)
    input_tile = make_tile(rect)
    tile_loader = get_tile_loader(make_tile_getter(input_tile), input_tile.rect, window=(128, 128))
    mask = run_inference(tile_loader, one_channel_passthrough)
    assert mask.image.shape == input_tile.image.shape[:2]
    assert mask.rect == rect


def test_applies_a_max_pooling_model():
    rect = Rectangle(upper_left=(0, 0), width=256, height=256)
    input_tile = make_tile(rect)
    tile_loader = get_tile_loader(make_tile_getter(input_tile), input_tile.rect, window=(128, 128))
    mask = run_inference(tile_loader, downscaling_model)
    assert mask.image.shape == input_tile.image.shape[:2]
    assert mask.rect == rect


def test_applies_a_model_with_batching():
    rect = Rectangle(upper_left=(0, 0), width=256, height=256)
    input_tile = make_tile(rect)
    tile_loader = get_tile_loader(make_tile_getter(input_tile), input_tile.rect, window=(128, 128))
    mask = run_inference(tile_loader, downscaling_model, batch_size=2)
    assert mask.image.shape == input_tile.image.shape[:2]
    assert mask.rect == rect


def test_applies_a_model_with_one_big_batch():
    rect = Rectangle(upper_left=(0, 0), width=256, height=256)
    input_tile = make_tile(rect)
    tile_loader = get_tile_loader(make_tile_getter(input_tile), input_tile.rect, window=(128, 128))
    mask = run_inference(tile_loader, downscaling_model, batch_size=4)
    assert mask.image.shape == input_tile.image.shape[:2]
    assert mask.rect == rect


MODEL_PATH = "tests/glomeruli_segmentation_16934_best_metric.model-384e1332.pth"


def test_applies_real_model():
    width, height = (2048, 2048)
    rect = Rectangle(upper_left=(0, 0), width=width, height=height)
    image = np.arange(width * height * 3, dtype=np.float32).reshape((width, height, 3))
    input_tile = Tile(image=image, rect=rect)
    tile_loader = get_tile_loader(make_tile_getter(input_tile), input_tile.rect, window=(1024, 1024))

    model = nn.Sequential(load_unet(MODEL_PATH), nn.Softmax(dim=1), SingleChannelPassthrough(channel=1))

    mask = run_inference(tile_loader, model)
    assert mask.image.shape == input_tile.image.shape[:2]
    assert mask.rect == rect
