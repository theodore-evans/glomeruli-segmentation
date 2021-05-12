import argparse
import logging

logger = logging.getLogger(__name__)

try:
    from app.api import API
except KeyError:
    logger.warning("No EMPAIA API available, using mock API")
    from app.mock_api import MockAPI as API

from app.entity_extractor import EntityExtractor
from app.inference_runner import InferenceRunner
from app.output_serializer import result_to_collection
from data.preprocessing import raw_test_transform
from data.wsi_tile_fetcher import WSITileFetcher

DEFAULT_MODEL_PATH = "../model/hacking_kidney_16934_best_metric.model-384e1332.pth"


parser = argparse.ArgumentParser(description="Detect glomeruli on kidney wsi")
parser.add_argument("--model", dest="model_path", default=DEFAULT_MODEL_PATH)
args = parser.parse_args()

inference = InferenceRunner(args.model_path, data_transform=raw_test_transform())
api = API()

my_rectangle = api.get_input("my_rectangle")
kidney_wsi = api.get_input("kidney_wsi")
upper_left = my_rectangle["upper_left"]
size_to_process = (my_rectangle["width"], my_rectangle["height"])


def tile_request(rect):
    return api.get_wsi_tile(kidney_wsi, rect)


tile_fetcher = WSITileFetcher(tile_request, my_rectangle)
output_mask = inference(tile_fetcher)

entity_extractor = EntityExtractor(upper_left=upper_left)
extracted_entities = entity_extractor.extract_entities(output_mask)

count_result = {"name": "Glomerulus Count", "type": "integer", "value": extracted_entities.count()}

api.post_output(key="glomerulus_count", data=count_result)

contour_result = result_to_collection(extracted_entities, kidney_wsi["id"], my_rectangle["id"])

api.post_output(key="glomeruli_polygons", data=contour_result)

api.put_finalize()
