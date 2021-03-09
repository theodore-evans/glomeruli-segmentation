import torch
import torch.nn
import cv2

import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image

from nn import load_model
from nn.segnet import SegNet
from data import Dataset

class InferenceRunner:
    def __init__(self, model_path: str) -> None:
        
        # These values are calculated for the HuBMAP test dataset
        TORCH_VISION_MEAN = np.asarray([0.485, 0.456, 0.406])
        TORCH_VISION_STD = np.asarray([0.229, 0.224, 0.225])
        self.test_transform = Compose([ToTensor(), Normalize(mean=TORCH_VISION_MEAN, std=TORCH_VISION_STD)])

        self.model = load_model(model_path)

    def load_model(self, model_path: str) -> SegNet:
        torch.set_grad_enabled(False)
        model = load_model(model_path)
        model = model.cuda().eval()
        return model

    def run_inference_on_entry(self, entry: np.ndarray) -> np.ndarray:
        inputs = self.test_transform(input)
        H, W = inputs.shape[:2]
        inputs = inputs.unsqueeze(0)

        probs = torch.nn.functional.softmax(self.model(inputs.cuda())).cpu()[0, 1, :, :].numpy()
        prediction = (cv2.resize(probs, (H, W), interpolation=cv2.INTER_LINEAR) * 255).astype(dtype=np.uint8)
        return prediction

    def run_inference_on_dataset(self, dataset: Dataset):# -> np.ndarray:
        pass