import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from data import (
    KidneyValidDataset,
    create_dataset,
    create_test_datasets,
    list_tiff_files,
)
from nn import load_model
from torchvision.transforms import Compose, Normalize, ToTensor

st.title("Hacking Kidney Demo")

parser = argparse.ArgumentParser()
parser.add_argument("--image-size", default=1024, type=int, help="crop image window size")
parser.add_argument("--resize", default=256, type=int, help="resize image before forward")
parser.add_argument("--data-fold", default=0, type=int, help="crop image window size")
parser.add_argument(
    "--data-root", default="/data/hubmap-kidney-segmentation", type=str, help="path to dataset",
)
parser.add_argument(
    "--mode", default="valid", choices=["train", "valid", "test"], help="path to a model",
)
parser.add_argument("--model", default=None, type=str, help="path to a model")
parser.add_argument("--submission", default=None, type=str, help="path to a submission")
args = parser.parse_args()
print(args)

TORCH_VISION_MEAN = np.asarray([0.485, 0.456, 0.406])
TORCH_VISION_STD = np.asarray([0.229, 0.224, 0.225])
test_transform = Compose([ToTensor(), Normalize(mean=TORCH_VISION_MEAN, std=TORCH_VISION_STD)])

if args.submission:
    files = list_tiff_files(Path(os.path.join(args.data_root, "test")))
    submission_csv = pd.read_csv(args.submission)

    def mask_encoding(f):
        return submission_csv.loc[submission_csv["id"] == f.stem]["predicted"].values[0]

    datasets = [KidneyValidDataset(f, mask_encoding(f), window_size=args.image_size) for f in files]
    dataset = sum(datasets)
else:
    if args.mode == "test":
        dataset = create_test_datasets(os.path.join(args.data_root, "test"), image_size=args.image_size)
    else:
        dataset = create_dataset(
            data_root=args.data_root, mode=args.mode, data_fold=args.data_fold, image_size=args.image_size,
        )
print(dataset)

model = None
if args.model:
    torch.set_grad_enabled(False)
    model = load_model(args.model)
    model = model.cuda().eval()


def mask_to_rgb(mask):
    rgb = np.ones((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
    mask_3 = ((mask & 2 ** 3) > 0).astype(np.float32)
    rgb[..., 0] = (((mask & 2 ** 0) > 0).astype(np.float32) + mask_3) / 2
    rgb[..., 1] = (((mask & 2 ** 1) > 0).astype(np.float32) + mask_3) / 2
    rgb[..., 2] = (mask & 2 ** 2) > 0
    return 1 - rgb


def get_image(entry):
    img = np.array(entry["input"])
    H, W = img.shape[:2]
    mask = np.zeros_like(img)
    if "mask" in entry:
        mask[entry["mask"] > 0, -1] = 255
    else:
        mask[:, :, :] = 0
    alpha = 0.5

    if model is not None:
        inputs = test_transform(img)
        print(inputs)
        inputs = inputs.unsqueeze(0)
        probs = torch.nn.functional.softmax(model(inputs.cuda())).cpu()[0, 1, :, :].numpy()
        prediction = (cv2.resize(probs, (H, W), interpolation=cv2.INTER_LINEAR) * 255).astype(dtype=np.uint8)
        mask[:, :, 1] = prediction
    return cv2.addWeighted(img, alpha, mask, (1 - alpha), 0)


sample_id = st.sidebar.selectbox("Please select one sample", range(len(dataset)), index=149)
if st.sidebar.button("random"):
    sample_id = np.random.choice(len(dataset))

st.image(get_image(dataset[sample_id]), caption=f"data {sample_id}")
