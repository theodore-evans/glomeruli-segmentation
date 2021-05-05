from app.api import API
from app.entity_extractor import EntityExtractor
from app.inference_runner import InferenceRunner
from app.mock_api import MockAPI
from app.serializer import contours_to_collection
from data.preprocessing import raw_test_transform
from data.wsi_tile_fetcher import WSITileFetcher

MODEL_PATH = "./model/hacking_kidney_16934_best_metric.model-384e1332.pth"
inference = InferenceRunner(MODEL_PATH, data_transform=raw_test_transform())
api = MockAPI()

my_rectangle = api.get_input("my_rectangle")
kidney_wsi = api.get_input("kidney_wsi")
upper_left = my_rectangle["upper_left"]
size_to_process = (my_rectangle["width"], my_rectangle["height"])


def tile_request(rect):
    return api.get_wsi_tile(kidney_wsi, rect)


tile_fetcher = WSITileFetcher(tile_request, my_rectangle)
output_mask = inference(tile_fetcher)

entity_extractor = EntityExtractor(upper_left=upper_left)
contours = entity_extractor.extract_contours(output_mask)
count = entity_extractor.count_entities(contours)

count_result = {"name": "Glomerulus Count", "type": "integer", "value": count}
api.post_output(key="glomerulus_count", data=count_result)

contour_result = contours_to_collection(contours, kidney_wsi["id"], my_rectangle["id"])

api.post_output(key="glomeruli_polygons", data=contour_result)

api.put_finalize()
