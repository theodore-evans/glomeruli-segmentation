import torch
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
from nn import load_model
from nn.segnet import SegNet

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