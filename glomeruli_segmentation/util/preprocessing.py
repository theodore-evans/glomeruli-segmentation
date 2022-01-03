import numpy as np
from torchvision.transforms.transforms import Compose, Normalize, ToTensor


def kaggle_test_transform() -> Compose:
    TORCH_VISION_MEAN = np.array([0.485, 0.456, 0.406])
    TORCH_VISION_STD = np.array([0.229, 0.224, 0.225])
    transform = Compose(Normalize(mean=TORCH_VISION_MEAN, std=TORCH_VISION_STD))
    return transform


def raw_test_transform() -> Compose:
    TORCH_VISION_MEAN = np.asarray([0.485, 0.456, 0.406])
    TORCH_VISION_STD = np.asarray([0.229, 0.224, 0.225])
    test_transform = Compose([ToTensor(), Normalize(mean=TORCH_VISION_MEAN, std=TORCH_VISION_STD)])
    return test_transform
