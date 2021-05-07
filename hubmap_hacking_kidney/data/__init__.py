import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from .dataset import Dataset
from .rle import rle_decode
from .tifffile_dataset import KidneyDataset


def prob_to_mask(prob, threshold=0.5):
    return prob > threshold


def pred_to_prob(predictions, height, width, ignore_border=0):
    """merge prediction to one prob array"""
    prob = torch.zeros((height, width), dtype=torch.float32)  # type: ignore
    n = torch.zeros((height, width), dtype=torch.uint8)  # type: ignore
    for image_id, pred in predictions:
        x, y = [int(i) for i in image_id.split("_")]

        w = pred.shape[1]
        h = pred.shape[0]
        xx = x + w
        yy = y + h
        l = 0
        t = 0

        if ignore_border > 0:
            if x > 0:
                x += ignore_border
                l = ignore_border
            if xx < width:
                xx -= ignore_border
            if y > 0:
                y += ignore_border
                t = ignore_border
            if yy < height:
                yy -= ignore_border
            w = xx - x + l
            h = yy - y + t
        prob[y:yy, x:xx] += pred[t:h, l:w]
        n[y:yy, x:xx] += 1

    # if not n.all():
    #     print(np.transpose(np.nonzero(n==0)))

    prob /= n
    return prob


class KidneyTrainDataset(KidneyDataset):
    classnames = ["background", "glomeruli"]

    def __init__(self, tiff_file, mask_encoding, window_size=1024, image_stride=None, scale=None):
        super().__init__(tiff_file, scale=scale)
        self.tiff_file = tiff_file
        self.original_mask_file = None
        self.original_mask = self.mask = None
        if mask_encoding:
            self.original_mask = self.mask = rle_decode(
                mask_encoding, (self.original_height, self.original_width)
            )
            if scale:
                self.mask = cv2.resize(
                    self.mask,
                    (self.width, self.height),
                    interpolation=cv2.INTER_NEAREST,
                )

        self.window_size = window_size
        self.image_stride = image_stride or window_size
        self.cols = self.width - self.window_size
        self.rows = self.height - self.window_size

    def __len__(self):
        return self.width // self.window_size * self.height // self.window_size

    def __repr__(self):
        fmt_str = super().__repr__()
        fmt_str += self.repr_indent + f"implementation: {KidneyDataset.__module__}\n"
        fmt_str += self.repr_indent + f"file: {self.tiff_file}\n"
        fmt_str += (
            self.repr_indent
            + f"original_width x original_height: {self.original_width} x {self.original_height}\n"
        )
        fmt_str += self.repr_indent + f"width x height: {self.width} x {self.height}\n"
        fmt_str += self.repr_indent + f"window_size: {self.window_size}\n"
        fmt_str += self.repr_indent + f"image_stride: {self.image_stride}\n"
        return fmt_str

    def getitem(self, index):
        x = random.randint(0, self.cols)
        y = random.randint(0, self.rows)

        img = self.read(x, y, self.window_size, self.window_size)
        mask = self.get_mask(x, y, self.window_size, self.window_size)
        return dict(input=Image.fromarray(img), mask=mask)

    def get_mask(self, x, y, width, height):
        return self.mask[y : y + height, x : x + width]


class KidneyTestDataset(KidneyTrainDataset):
    def __init__(self, tiff_file, mask_encoding, window_size=1024, image_stride=None, scale=None):
        super().__init__(
            tiff_file,
            mask_encoding,
            window_size=window_size,
            image_stride=image_stride,
            scale=scale,
        )
        if window_size > self.width or window_size > self.height:
            raise RuntimeError(
                f"{tiff_file} window_size ({window_size}) is bigger than image size ({self.height} x {self.width})"
            )

        original_mask_file = tiff_file.parent / (tiff_file.stem + "_orig_mask.png")
        if original_mask_file.exists():
            self.original_mask = cv2.imread(str(original_mask_file), cv2.IMREAD_UNCHANGED)
            self.original_mask_file = original_mask_file
            self.original_height = self.original_mask.shape[0]
            self.original_width = self.original_mask.shape[1]

        self.offset_x = list(range(0, self.width - self.window_size + self.image_stride, self.image_stride))
        self.offset_y = list(range(0, self.height - self.window_size + self.image_stride, self.image_stride))

        # make sure the cropped images are inside, so no padding needed
        self.offset_x[-1] = self.shift_offset_into_image(self.offset_x[-1], self.width)
        self.offset_y[-1] = self.shift_offset_into_image(self.offset_y[-1], self.height)

    def shift_offset_into_image(self, x, max_x):
        if x + self.window_size > max_x:
            x = max_x - self.window_size
        return x

    def __len__(self):
        return len(self.offset_x) * len(self.offset_y)

    def __repr__(self):
        fmt_str = super().__repr__()
        fmt_str += self.repr_indent + f"X x Y: {len(self.offset_x)} x {len(self.offset_y)}\n"
        fmt_str += self.repr_indent + f"original_mask_file: {self.original_mask_file}\n"
        return fmt_str

    def get_offset(self, index):
        width = len(self.offset_x)
        yi = index // width
        xi = index % width
        x = self.offset_x[xi]
        y = self.offset_y[yi]
        return x, y

    def pad_to_window_size(self, array):
        height, width = array.shape[:2]
        if height < self.window_size or width < self.window_size:
            pad_bottom = self.window_size - height
            pad_right = self.window_size - width
            padding = ((0, pad_bottom), (0, pad_right), (0, 0))
            return np.pad(array, padding[: array.ndim])
        return array

    def getitem(self, index):
        x, y = self.get_offset(index)
        img = self.read(x, y, self.window_size, self.window_size)
        return dict(input=Image.fromarray(img), image_id=f"{x}_{y}")

    def pred_to_mask(self, predictions, threshold=0.5, ignore_border=0):
        prob = pred_to_prob(
            predictions,
            height=self.height,
            width=self.width,
            ignore_border=ignore_border,
        )  # height x width
        if self.original_height != self.height or self.original_width != self.width:
            prob = prob[None, None]  # mini-batch x channels x height x width
            prob = F.interpolate(
                prob,
                (self.original_height, self.original_width),
                mode="bilinear",
                align_corners=False,
            )
            prob = prob.squeeze()  # height x width
        mask = prob_to_mask(prob, threshold=threshold)
        # if self.original_height != self.height or self.original_width != self.width:
        #     mask = mask.numpy().astype(np.uint8)
        #     mask = cv2.resize(mask, (self.original_width, self.original_height), interpolation=cv2.INTER_NEAREST)
        #     mask = torch.from_numpy(mask.astype(np.bool))

        return mask


class KidneyValidDataset(KidneyTestDataset):
    def getitem(self, index):
        x, y = self.get_offset(index)
        img = self.read(x, y, self.window_size, self.window_size)
        mask = self.get_mask(x, y, self.window_size, self.window_size)

        return dict(input=Image.fromarray(img), mask=mask)


def list_tiff_files(data_root):
    tiff_files = [f for f in data_root.iterdir() if f.suffix == ".tiff"]
    return sorted(tiff_files)


def create_dataset(
    data_root,
    mode,
    data_fold=0,
    image_size=1024,
    image_stride=None,
    scale=None,
    transform=None,
):
    data_root = Path(data_root)
    tiff_files = list_tiff_files(data_root / "train")

    train_csv = pd.read_csv(data_root / "train.csv")

    def mask_encoding(f):
        return train_csv.loc[train_csv["id"] == f.stem]["encoding"].values[0]

    if mode == "train":
        tiff_files = [f for i, f in enumerate(tiff_files) if i != data_fold]
        datasets = [
            KidneyTrainDataset(
                f,
                mask_encoding(f),
                window_size=image_size,
                image_stride=image_stride,
                scale=scale,
            )
            for f in tiff_files
        ]
        dataset = sum(datasets)
    elif mode in ("valid", "test"):
        datasetT = KidneyValidDataset if mode == "valid" else KidneyTestDataset
        if data_fold >= 0:
            f = tiff_files[data_fold]
            dataset = datasetT(
                f,
                mask_encoding(f),
                window_size=image_size,
                image_stride=image_stride,
                scale=scale,
            )
        else:
            datasets = [
                datasetT(
                    f,
                    mask_encoding(f),
                    window_size=image_size,
                    image_stride=image_stride,
                    scale=scale,
                )
                for f in tiff_files
            ]
            dataset = sum(datasets)
    else:
        raise NotImplementedError(mode)

    if transform:
        dataset = dataset >> transform
    return dataset


def create_test_datasets(test_folder_or_file, image_size=1024, image_stride=None, scale=None, transform=None):
    test_folder_or_file = Path(test_folder_or_file)
    if test_folder_or_file.is_dir():
        files = list_tiff_files(test_folder_or_file)
    else:
        files = [test_folder_or_file]

    for f in files:
        dataset = KidneyTestDataset(f, None, window_size=image_size, image_stride=image_stride, scale=scale)

        if transform:
            dataset = dataset >> transform

        yield dataset
