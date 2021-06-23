import math
import warnings

import torch
import torch.nn.functional as F
from torch import nn


def tta_pre_process(inputs, tta):
    if tta & 1:
        inputs = inputs.flip(-1)
    if tta & 2:
        inputs = inputs.flip(-2)
    if tta & 4:
        inputs = inputs.transpose(-1, -2)
    return inputs


def tta_post_process(outputs, tta):
    if tta & 4:
        outputs = outputs.transpose(-1, -2)
    if tta & 2:
        outputs = outputs.flip(-2)
    if tta & 1:
        outputs = outputs.flip(-1)
    return outputs


def divisible_padding(size, size_divisible):
    """return size to pad for divisible"""
    return int(math.ceil(size / size_divisible) * size_divisible) - size


def sliding_window(size, window_size, stride):
    if window_size >= size:
        yield 0, size
    else:
        for i in range(0, size, stride):
            ii = i + window_size
            if ii > size:
                ii = size
                i = size - window_size
            yield i, ii


class SegNet(nn.Module):
    MAX_PREDICT_WINDOW = float("inf")
    DIVISIBLE_PAD = 1

    def __init__(self, objectness=False, tta=0, scales=None, resize=None):
        super().__init__()
        if scales and resize:
            resize = None
            warnings.warn(
                f"testing scales = {scales}, please make sure input matches training resize = {resize}"
            )

        self.objectness = objectness
        self.tta = tta
        self.scales = scales if scales else [1]
        self.resize_input = resize

    def predict(self, x):
        probs = self.predict_probs_window(x, self.MAX_PREDICT_WINDOW)
        probs = probs[:, 1]  # take class 1 only
        if x.dim() == 4:  # batch inference or not
            probs = probs.squeeze(1)
        else:
            probs = probs.squeeze()

        return probs.cpu()

    def predict_probs_window(self, x, window_size, stride=None):
        width = x.shape[-1]
        height = x.shape[-2]

        if width <= window_size and height <= window_size:
            return self.predict_probs(x)

        if stride is None:
            stride = window_size // 2

        y = None
        c = None
        for i, ii in sliding_window(height, window_size, stride):
            for j, jj in sliding_window(width, window_size, stride):
                img = x[..., i:ii, j:jj]

                probs = self.predict_probs(img)
                if y is None:
                    y = probs.new_zeros((probs.shape[0], probs.shape[1], height, width))
                    c = probs.new_zeros((probs.shape[0], probs.shape[1], height, width))

                y[..., i:ii, j:jj] += probs
                c[..., i:ii, j:jj] += 1
        y /= c

        return y

    def predict_probs(self, x):
        if x.dim() < 4:
            x = x.unsqueeze(0)
        param = next(self.parameters())
        x = x.to(param)

        height, width = x.shape[-2:]
        probs = 0
        for scale in self.scales:
            scaled_image = F.interpolate(
                x, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=True,
            )
            scaled_pred = self.predict_tta(scaled_image)
            pred = F.interpolate(scaled_pred, (height, width), mode="bilinear", align_corners=False)
            probs += pred
        probs /= len(self.scales)

        return probs

    def predict_tta(self, x):
        if self.tta >= 0:
            # process once
            y = self.forward_tta(x, self.tta)
        else:
            # process 8 times
            max_tta = -self.tta
            y = 0
            for t in range(max_tta):
                y += self.forward_tta(x, t)
            y /= max_tta

        return y

    def forward_tta(self, x, tta):
        x = tta_pre_process(x, tta)
        y = self(x)
        if isinstance(y, dict):
            y = y["out"]
        y = y.softmax(1)
        y = tta_post_process(y, tta)
        return y

    def forward(self, x):
        if self.resize_input:
            x = F.interpolate(x, self.resize_input, mode="bilinear", align_corners=False)

        pad_bottom = divisible_padding(x.shape[-2], self.DIVISIBLE_PAD)
        pad_right = divisible_padding(x.shape[-1], self.DIVISIBLE_PAD)
        if pad_bottom > 0 or pad_right > 0:
            x = F.pad(x, [0, pad_right, 0, pad_bottom])

        y = self._forward(x)

        if pad_bottom > 0 or pad_right > 0:
            y = y[..., : y.shape[-2] - pad_bottom, : y.shape[-1] - pad_right]

        if self.objectness:
            y = torch.sigmoid(y)
            objectness = 1 - y[:, :1]  # prob of foreground
            cls = y[:, 1:] * objectness
            y = torch.cat((y[:, :1], cls), dim=1)

        return y

    def _forward(self, x):
        raise NotImplementedError(self.__class__)
