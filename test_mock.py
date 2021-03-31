# %%
from app.entity_extractor import EntityExtractor
from app.inference_runner import InferenceRunner
from data.wsi_tile_fetcher import WSITileFetcher
from app.mock_api import MockAPI
from app.serializer import contours_to_collection
from data.preprocessing import raw_test_transform


MODEL_PATH = "./model/hacking_kidney_16934_best_metric.model-384e1332.pth"
inference = InferenceRunner(MODEL_PATH, data_transform=raw_test_transform())
# %%
mock_api = MockAPI()
# %%
my_rectangle = mock_api.get_input("my_rectangle")
kidney_wsi = mock_api.get_input("kidney_wsi")
upper_left = my_rectangle["upper_left"]
size_to_process = (my_rectangle["width"], my_rectangle["height"])


def tile_request(rect): return mock_api.get_wsi_tile(kidney_wsi, rect)


tile_fetcher = WSITileFetcher(tile_request, my_rectangle)
# %%
output_mask = inference(tile_fetcher)

# TODO get rid of 3 channels for output_mask
entity_extractor = EntityExtractor(upper_left=upper_left)
contours = entity_extractor.extract_contours(output_mask)
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

mock_api.put_finalize()
# %%
