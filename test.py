# %%
from app.entity_extractor import EntityExtractor
import numpy as np
from app.inference_runner import InferenceRunner
from data.wsi_tile_fetcher import WSITileFetcher
from app.mock_api import MockAPI
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
from app.serializer import contours_to_collection

from app.api import API
from app.data_types import Rectangle

TORCH_VISION_MEAN = np.asarray([0.485, 0.456, 0.406])
TORCH_VISION_STD = np.asarray([0.229, 0.224, 0.225])
test_transform = Compose([ToTensor(), Normalize(
    mean=TORCH_VISION_MEAN, std=TORCH_VISION_STD)])

model_path = "/model/hacking_kidney_16934_best_metric.model-384e1332.pth"
sample_image_file = "/data/hubmap-kidney-segmentation/train/54f2eec69.tiff"
inference = InferenceRunner(model_path, data_transform=test_transform)
# %%
mock_api = MockAPI(sample_image_file)
# %%


# my_rectangle: Rectangle = {"upper_left": upper_left,
#                           "width": size_to_process[0], "height": size_to_process[1], "level": 0}
my_rectangle = mock_api.get_input("my_rectangle")
kidney_wsi = mock_api.get_input("kidney_wsi")
upper_left = my_rectangle["upper_left"]
size_to_process = (my_rectangle["width"], my_rectangle["height"])
def tile_request(rect): return mock_api.get_wsi_tile(kidney_wsi, rect)


tile_fetcher = WSITileFetcher(tile_request, my_rectangle)
# %%
input_image = np.transpose(np.uint8(
    mock_api.image_data[:, upper_left[0]:upper_left[0]+size_to_process[0], upper_left[1]:upper_left[1] + size_to_process[1]]), (1, 2, 0))
output_mask = np.zeros_like(input_image)
output_mask[:, :, 1] = inference(tile_fetcher)

# TODO get rid of 3 channels for output_mask
entity_extractor = EntityExtractor(upper_left=upper_left)
contours = entity_extractor.extract_contours(output_mask[:, :, 1])
count = entity_extractor.count_entities(contours)

# %%
count_result = {
    "name": "Glomerulus Count",
    "type": "integer",
    "value": count
}
mock_api.post_output(key="glomerulus_count", data=count_result)

contour_result = contours_to_collection(
    contours, kidney_wsi["id"], my_rectangle["id"])

contour_result.items = [p.__dict__ for p in contour_result.items]

mock_api.post_output(key="glomeruli_polygons", data=contour_result.__dict__)

# %%
