import argparse
import functools
import os

import torch.nn as nn
from api_interface import ApiInterface
from entity_extractor import get_contours_from_mask
from inference import SingleChannelPassthrough, load_unet, run_inference
from logging_tools import get_log_level, get_logger
from output_serialization import serialize_result_to_collection
from tile_loader import get_tile_loader

BATCH_SIZE = 16


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

    api = ApiInterface(api_url, job_id, headers, logger=get_logger("api", app_log_level))
    roi = api.get_rectangle("region_of_interest")
    slide = api.get_wsi("slide")

    tile_request = functools.partial(api.get_wsi_tile, slide=slide)
    tile_loader = get_tile_loader(tile_request, roi, window=(1024, 1024))

    model = nn.Sequential(load_unet(model_path), nn.Softmax(dim=1), SingleChannelPassthrough(channel=1))
    segmentation_mask = run_inference(tile_loader, model, batch_size=BATCH_SIZE)
    glomeruli_contours, _ = get_contours_from_mask(segmentation_mask)

    number_of_glomeruli = {"name": "Glomerulus Count", "type": "integer", "value": len(glomeruli_contours)}
    api.post_output(key="glomerulus_count", data=number_of_glomeruli)

    annotations = serialize_result_to_collection(glomeruli_contours, slide, roi)
    api.post_output(key="glomeruli_polygons", data=annotations)

    api.put_finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect glomeruli on kidney wsi")
    parser.add_argument("-v", "--verbose", help="increase logging verbosity", action="count", default=0)
    args = parser.parse_args()

    main(verbosity=args.verbose)
