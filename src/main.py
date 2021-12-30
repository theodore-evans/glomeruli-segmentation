import argparse
import os

from app.api_interface import ApiInterface
from app.data_types import TileGetter
from app.entity_extractor import EntityExtractor
from app.inference import InferenceRunner
from app.logging_tools import get_log_level, get_logger
from app.serialization import result_to_collection
from app.tile_loader import TileLoader, get_tile_loader

DEFAULT_MODEL_PATH = "../model/hacking_kidney_16934_best_metric.model-384e1332.pth"


def main(model_path: str, verbosity: int):
    app_log_level = get_log_level(verbosity)
    logger = get_logger("main", app_log_level)

    try:
        api_url = os.environ["EMPAIA_APP_API"]
        job_id = os.environ["EMPAIA_JOB_ID"]
        headers = {"Authorization": f"Bearer {os.environ['EMPAIA_TOKEN']}"}
    except KeyError as e:
        logger.error("Missing EMPAIA API environment variables")
        raise e

    api = ApiInterface(verbosity, api_url, job_id, headers, logger=get_logger("api", app_log_level))

    roi = api.get_input("region_of_interest")
    slide = api.get_input("kidney_wsi")
    roi_origin = roi["upper_left"]

    tile_request: TileGetter = lambda x: api.get_wsi_tile(slide, x)
    tile_loader = get_tile_loader(tile_request, roi, window=(1024,1024))
                                  
    inference_runner = InferenceRunner(model_path, logger=get_logger("inference", app_log_level))
    output_mask = inference_runner(tile_loader)

    entity_extractor = EntityExtractor(origin=roi_origin)
    extracted_entities = entity_extractor.extract_from_mask(output_mask)

    count_result = {"name": "Glomerulus Count", "type": "integer", "value": extracted_entities.count()}

    api.post_output(key="glomerulus_count", data=count_result)

    contour_result = result_to_collection(extracted_entities, slide["id"], roi["id"])

    api.post_output(key="glomeruli_polygons", data=contour_result)

    api.put_finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect glomeruli on kidney wsi")

    parser.add_argument("--model", dest="model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("-v", "--verbose", help="increase logging verbosity", action="count", default=0)

    args = parser.parse_args()

    main(args.model, args.verbose)
