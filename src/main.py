import argparse
import functools
import os

import torch
import torch.nn as nn
from app.api_interface import ApiInterface
from app.entity_extractor import EntityExtractor
from app.inference import run_inference
from app.logging_tools import get_log_level, get_logger
from app.serialization import result_to_collection
from app.tile_loader import get_tile_loader


def main(verbosity: int):
    app_log_level = get_log_level(verbosity)
    logger = get_logger("main", app_log_level)

    try:
        api_url = os.environ["EMPAIA_APP_API"]
        job_id = os.environ["EMPAIA_JOB_ID"]
        headers = {"Authorization": f"Bearer {os.environ['EMPAIA_TOKEN']}"}
        model_path = os.environ["MODEL_PATH"]
    except KeyError as e:
        logger.error("Missing EMPAIA API environment variables")
        raise e

    api = ApiInterface(verbosity, api_url, job_id, headers, logger=get_logger("api", app_log_level))

    roi = api.get_rectangle("region_of_interest")
    slide = api.get_wsi("kidney_wsi")

    tile_request = functools.partial(api.get_wsi_tile, slide=slide)
    tile_loader = get_tile_loader(tile_request, roi, window=(1024, 1024))

    model = nn.Sequential(torch.load(model_path, map_location="cpu"), nn.Softmax(dim=1))
    output_mask = run_inference(tile_loader, model, batch_size=16)

    # TODO: make this a function
    entity_extractor = EntityExtractor(origin=roi.upper_left)
    extracted_entities = entity_extractor.extract_from_mask(output_mask)

    count_result = {"name": "Glomerulus Count", "type": "integer", "value": extracted_entities.count()}

    api.post_output(key="glomerulus_count", data=count_result)

    contour_result = result_to_collection(extracted_entities, slide["id"], roi["id"])

    api.post_output(key="glomeruli_polygons", data=contour_result)

    api.put_finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect glomeruli on kidney wsi")

    parser.add_argument("-v", "--verbose", help="increase logging verbosity", action="count", default=0)

    args = parser.parse_args()

    main(args.verbose)
