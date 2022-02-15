import argparse
import functools
import os

import torch.nn as nn

from glomeruli_segmentation.api_interface import ApiInterface
from glomeruli_segmentation.data_classes import Rectangle, Wsi
from glomeruli_segmentation.extract_contours import get_contours_from_mask
from glomeruli_segmentation.logging_tools import get_log_level, get_logger
from glomeruli_segmentation.output_serialization import (
    classifications_to_collection,
    confidences_to_collection,
    contours_to_collection,
)
from glomeruli_segmentation.run_inference import SingleChannelPassthrough, load_unet, run_inference_on_tiles
from glomeruli_segmentation.tile_loader import get_tile_loader
from glomeruli_segmentation.util.preprocessing import get_kaggle_test_transform

TRANSFORM = get_kaggle_test_transform()
EAD_NAMESPACE = "org.empaia.dai.glomeruli_segmentation.v1"


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

    logger.info(f"EMPAIA_JOB_ID: {job_id}")

    api = ApiInterface(api_url, job_id, headers, logger=get_logger("api", app_log_level))
    roi = api.get_input("region_of_interest", Rectangle)
    slide = api.get_input("slide", Wsi)
    tile_request = functools.partial(api.get_wsi_tile, slide=slide)
    tile_loader = get_tile_loader(tile_request, roi, window=(1024, 1024))

    model = nn.Sequential(load_unet(model_path), nn.Softmax(dim=1), SingleChannelPassthrough(channel=1))
    segmentation_mask = run_inference_on_tiles(tile_loader, model, transform=TRANSFORM)
    glomeruli_contours, confidence_values = get_contours_from_mask(segmentation_mask)

    number_of_glomeruli = {"name": "Glomeruli Count", "type": "integer", "value": len(glomeruli_contours)}
    api.post_output(key="glomeruli_count", data=number_of_glomeruli)

    annotations = contours_to_collection(glomeruli_contours, slide, roi)
    annotations = api.post_output(key="glomeruli", data=annotations)

    confidences = confidences_to_collection(annotations, confidence_values)
    classifications = classifications_to_collection(
        annotations,
        confidence_values,
        pos_condition=lambda x: x > 0.5,
        pos_class=EAD_NAMESPACE + ".classes.glomerulus",
        neg_class=EAD_NAMESPACE + ".classes.anomaly",
    )

    api.post_output(key="glomeruli_confidences", data=confidences)
    api.post_output(key="glomeruli_classifications", data=classifications)

    api.put_finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect glomeruli on kidney wsi")
    parser.add_argument("-v", "--verbose", help="increase logging verbosity", action="count", default=0)
    args = parser.parse_args()

    main(verbosity=args.verbose)
