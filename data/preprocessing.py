import numpy as np
from torchvision.transforms.transforms import Normalize, Compose

def kaggle_test_transform() -> Compose:
    TORCH_VISION_MEAN = np.array([0.485, 0.456, 0.406])
    TORCH_VISION_STD = np.array([0.229, 0.224, 0.225])
    transform = Compose(Normalize(mean=TORCH_VISION_MEAN, std=TORCH_VISION_STD))
    return transform