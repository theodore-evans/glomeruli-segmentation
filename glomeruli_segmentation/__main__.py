import argparse
import functools
import os

import torch.nn as nn
from torchinfo import summary

from glomeruli_segmentation.api_interface import ApiInterface
from glomeruli_segmentation.data_classes import Rectangle, Wsi
from glomeruli_segmentation.extract_contours import get_contours_from_mask
from glomeruli_segmentation.logging_tools import get_log_level, get_logger
from glomeruli_segmentation.output_serialization import (
    classify_annotations,
    create_annotation_collection,
    link_result_details,
)
from glomeruli_segmentation.run_inference import SingleChannelPassthrough, load_unet, run_inference_on_tiles
from glomeruli_segmentation.tile_loader import get_tile_loader
from glomeruli_segmentation.util.preprocessing import get_kaggle_test_transform

# TODO: load from config
TENSOR_TRANSFORM = get_kaggle_test_transform()

EAD_NAMESPACE = "org.empaia.dai.glomeruli_segmentation.v1"
ANOMALY_THRESHOLD = 0.70
GLOMERULUS_CLASS = EAD_NAMESPACE + ".classes.glomerulus"
ANOMALY_CLASS = EAD_NAMESPACE + ".classes.anomaly"

WINDOW_SIZE = (1024, 1024)
STRIDE = (512, 512)
INPUT_SIZE = (1, 3, *WINDOW_SIZE)


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
    tile_loader = get_tile_loader(tile_request, roi, window=WINDOW_SIZE, stride=STRIDE)

    unet = load_unet(model_path)
    model = nn.Sequential(unet, nn.Softmax(dim=1), SingleChannelPassthrough(channel=1))
    summary(model, input_size=INPUT_SIZE, depth=1, verbose=verbosity)

    segmentation_mask = run_inference_on_tiles(tile_loader, model, transform=TENSOR_TRANSFORM)
    contours, confidences = get_contours_from_mask(segmentation_mask)
    classifications, glomeruli_count, _ = classify_annotations(
        confidences,
        condition=lambda x: x > ANOMALY_THRESHOLD,
        class_if_true=GLOMERULUS_CLASS,
        class_if_false=ANOMALY_CLASS,
    )

    annotations = create_annotation_collection(contours, slide, roi)
    annotations = api.post_output(key="glomeruli", data=annotations)

    confidence_collection, classification_collection = link_result_details(annotations, confidences, classifications)

    api.post_output(key="glomeruli_confidences", data=confidence_collection)
    api.post_output(key="glomeruli_classifications", data=classification_collection)
    api.post_output(
        key="glomeruli_count", data={"name": "Glomeruli Count", "type": "integer", "value": glomeruli_count}
    )

    api.put_finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect glomeruli on kidney wsi")
    parser.add_argument("-v", "--verbose", help="increase logging verbosity", action="count", default=0)
    args = parser.parse_args()

    main(verbosity=args.verbose)
