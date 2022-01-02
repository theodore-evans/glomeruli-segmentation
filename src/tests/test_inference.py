import numpy as np
import torch.nn as nn
from app.data_classes import Rectangle, Tile
from app.inference import run_inference
from app.tile_loader import get_tile_loader


def _make_tile(rect: Rectangle):
    width, height = (rect.width, rect.height)
    image = np.arange(width * height * 3, dtype=np.float32).reshape((width, height, 3))
    image = image * 255 / width * height * 3
    return Tile(image=image, rect=rect)


def _make_tile_getter(tile: Tile):
    def tile_getter(rect: Rectangle):
        x_start = rect.upper_left.x
        x_end = x_start + rect.width
        y_start = rect.upper_left.y
        y_end = y_start + rect.height
        return Tile(image=tile.image.copy()[x_start:x_end, y_start:y_end], rect=rect)

    return tile_getter


class OneChannelPassthrough(nn.Module):
    def forward(self, input):
        return input[:, 0:1, :, :]


one_channel_passthrough = OneChannelPassthrough()

downscaling_model = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1), OneChannelPassthrough())


def test_returns_a_mask_for_a_single_tile():
    rect = Rectangle(upper_left=(0, 0), width=128, height=128)
    input_tile = _make_tile(rect)
    tile_loader = iter([input_tile])

    mask = run_inference(tile_loader, one_channel_passthrough)
    assert isinstance(mask, Tile)
    assert mask.image.shape == input_tile.image.shape[:2]
    assert mask.rect == rect


def test_returns_a_combined_mask_for_split_tiles():
    rect = Rectangle(upper_left=(0, 0), width=256, height=256)
    input_tile = _make_tile(rect)
    tile_loader = get_tile_loader(_make_tile_getter(input_tile), input_tile.rect, window=(128, 128))
    mask = run_inference(tile_loader, one_channel_passthrough)
    assert mask.image.shape == input_tile.image.shape[:2]
    assert mask.rect == rect


def test_applies_a_max_pooling_model():
    rect = Rectangle(upper_left=(0, 0), width=256, height=256)
    input_tile = _make_tile(rect)
    tile_loader = get_tile_loader(_make_tile_getter(input_tile), input_tile.rect, window=(128, 128))
    mask = run_inference(tile_loader, downscaling_model)
    assert mask.image.shape == input_tile.image.shape[:2]
    assert mask.rect == rect


def test_applies_a_model_with_batching():
    rect = Rectangle(upper_left=(0, 0), width=256, height=256)
    input_tile = _make_tile(rect)
    tile_loader = get_tile_loader(_make_tile_getter(input_tile), input_tile.rect, window=(128, 128))
    mask = run_inference(tile_loader, downscaling_model, batch_size=2)
    assert mask.image.shape == input_tile.image.shape[:2]
    assert mask.rect == rect
    
def test_applies_a_model_with_one_big_batch():
    rect = Rectangle(upper_left=(0, 0), width=256, height=256)
    input_tile = _make_tile(rect)
    tile_loader = get_tile_loader(_make_tile_getter(input_tile), input_tile.rect, window=(128, 128))
    mask = run_inference(tile_loader, downscaling_model, batch_size=4)
    assert mask.image.shape == input_tile.image.shape[:2]
    assert mask.rect == rect