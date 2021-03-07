import api
import numpy as np
from nn import load_model
from nn.segnet import SegNet

from data.dataset import Dataset
from data.tifffile_dataset import KidneyDataset


def get_dataset_from_tile(tile: Image) -> Dataset:
    return KidneyDataset(tile)

def infer_segmentation_mask(dataset: Dataset, model: SegNet):
    

def count_glomeruli(dataset: Dataset) -> int:
    """
    Applies HuBMAP U-Net segmentation model to a selection wsi tile

    Parameters:
        wsi_tile: WSI image tile
    """
    return 42 

my_wsi = get_input("my_wsi") # 
my_rectangle = get_input("my_rectangle")

wsi_tile = get_wsi_tile(my_wsi, my_rectangle)

glomerulus_count = {
    "name": "glomerulus count",  # choose name freely
    "value": count_glomeruli(wsi_tile)
}

post_output("glomerulus_count", glomerulus_count)

put_finalize()
