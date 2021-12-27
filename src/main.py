import argparse
import logging
import os

from app.api_interface import ApiInterface
from app.entity_extractor import EntityExtractor
from app.inference_runner import InferenceRunner
from app.output_serializer import result_to_collection
from data.preprocessing import raw_test_transform
from data.wsi_tile_fetcher import WSITileFetcher

DEFAULT_MODEL_PATH = "../model/hacking_kidney_16934_best_metric.model-384e1332.pth"
VERBOSITY = 2 #Debug (temporary hard-coding)

def main(model_path: str):
    logger = logging.getLogger(__name__)
    query_parameters = dict()

    try:
        query_parameters.api_url = os.environ["EMPAIA_APP_API"]
        query_parameters.job_id = os.environ["EMPAIA_JOB_ID"]
        query_parameters.headers = {"Authorization": f"Bearer {os.environ['EMPAIA_TOKEN']}"}
    except KeyError as e:
        logger.error("Missing EMPAIA API environment variables")
        raise e

    api = ApiInterface(verbosity=VERBOSITY, parameters=query_parameters)

    inference_runner = InferenceRunner(model_path, data_transform=raw_test_transform())


    my_rectangle = api.get_input("my_rectangle")
    kidney_wsi = api.get_input("kidney_wsi")
    upper_left = my_rectangle["upper_left"]
    size_to_process = (my_rectangle["width"], my_rectangle["height"])


    def tile_request(rect):
        return api.get_wsi_tile(kidney_wsi, rect)


    tile_fetcher = WSITileFetcher(tile_request, my_rectangle)
    output_mask = inference_runner(tile_fetcher)

    entity_extractor = EntityExtractor(origin=upper_left)
    extracted_entities = entity_extractor.extract_from_mask(output_mask)

    count_result = {"name": "Glomerulus Count", "type": "integer", "value": extracted_entities.count()}

    api.post_output(key="glomerulus_count", data=count_result)

    contour_result = result_to_collection(extracted_entities, kidney_wsi["id"], my_rectangle["id"])

    api.post_output(key="glomeruli_polygons", data=contour_result)

    api.put_finalize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect glomeruli on kidney wsi")
    parser.add_argument("--model", dest="model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("-v", "--verbose", help="increase logging verbosity", action="count", default=0)

    args = parser.parse_args()
    
    main(args.model)