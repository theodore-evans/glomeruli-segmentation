import argparse
import asyncio
import os
from logging import INFO
from typing import Type

import torch
import torch.nn as nn
from padl import batch, transform, unbatch
from padl.transforms import Transform
from padl.utils import same
from torchvision import transforms as tvt

from glomeruli_segmentation.config import load_config

tvt = transform(tvt)
nn = transform(nn)

from glomeruli_segmentation.api_interface import ApiInterface
from glomeruli_segmentation.data_classes import Mask, Rectangle, Tile, Wsi
from glomeruli_segmentation.extract_results import get_results_from_mask
from glomeruli_segmentation.get_patches import get_patch_rectangles
from glomeruli_segmentation.logging_tools import get_log_level_for_verbosity, get_logger, log_level_names
from glomeruli_segmentation.model.unet import UNet
from glomeruli_segmentation.output_serialization import (
    create_annotation_collection,
    create_result_scalar,
    create_results_collection,
    link_results_by_id,
)


# TODO: add error reporting to /failure endpoint
async def main(verbosity: int):
    app_log_level = get_log_level_for_verbosity(verbosity, default_log_level=INFO)
    logger = get_logger("main", app_log_level)
    logger.info(f"Log level: {log_level_names[app_log_level]}")

    try:
        api_url = os.environ["EMPAIA_APP_API"]
        job_id = os.environ["EMPAIA_JOB_ID"]
        headers = {"Authorization": f"Bearer {os.environ['EMPAIA_TOKEN']}"}
        model_path = os.environ["MODEL_PATH"]
    except KeyError as e:
        logger.error(f"Missing environment variable: {e}")
        raise e

    logger.info(f"Using API URL: {api_url}")
    logger.info(f"Using job ID: {job_id}")
    logger.info(f"Using model path: {model_path}")

    config_path = os.environ.get("CONFIG_PATH")
    config = load_config(config_path, logger)

    pipeline: Transform = (
        tvt.ToTensor()
        >> tvt.Normalize(mean=config.torch_vision_mean, std=config.torch_vision_std)
        >> batch
        >> load_model_as_transform(model_path, UNet)
        >> nn.Softmax(dim=1)
        >> same[:, 1:2, :, :]
        >> unbatch
        >> tvt.Resize(config.window_size)
    )

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline.pd_to(device)
    except RuntimeError as runtime_error:
        logger.warning(f"Runtime error while loading pipeline to {device}: {runtime_error}")
        device = "cpu"
        pipeline.pd_to(device)

    logger.info(f"Using device: {device}")

    async with ApiInterface(api_url, job_id, headers, logger=get_logger("api", app_log_level)) as api:
        slide = await api.get_input(config.slide_key, Wsi)
        region_of_interest = await api.get_input(config.roi_key, Rectangle)

        patch_rectangles = get_patch_rectangles(region_of_interest, window=config.window_size, stride=config.stride)

        fetch_patch = lambda rect: api.get_wsi_tile(slide, rect)
        fetch_patch_tasks = [asyncio.create_task(fetch_patch(rect)) for rect in patch_rectangles]

        segmentation_mask = Mask.empty(region_of_interest)

        for task in asyncio.as_completed(fetch_patch_tasks):
            patch: Tile = await task
            logger.info(f"Processing patch {patch.rect}")
            patch.image = pipeline.infer_apply(patch.image)
            segmentation_mask.insert_patch(patch, blend_mode=config.blend_mode)

        results = get_results_from_mask(segmentation_mask, config=config)

        annotation_collection = create_annotation_collection(
            name="Glomerulus",
            slide=slide,
            roi=region_of_interest,
            annotation_type="polygon",
            values=results["contours"],
        )

        annotation_collection = await api.post_output(key=config.results_key_stub, data=annotation_collection)

        confidence_collection = create_results_collection(
            name="Confidence", item_type="float", values=results["confidences"]
        )
        classification_collection = create_results_collection(
            name=None, item_type="class", values=results["classifications"]
        )
        link_results_by_id(annotation_collection, [confidence_collection, classification_collection])

        await api.post_output(key=f"{config.results_key_stub}_confidences", data=confidence_collection)
        await api.post_output(key=f"{config.results_key_stub}_classifications", data=classification_collection)

        count_result = create_result_scalar(name="Count", item_type="integer", value=results["count"])
        await api.post_output(key=f"{config.results_key_stub}_count", data=count_result)

        await api.put_finalize()


def load_model_as_transform(model_path: str, model_class: Type) -> Transform:
    """
    Loads a model from a file and returns a padl.Transform that applies it.

    :param model_path: Path to the model file.
    :param model_class: Class of the model.
    """
    model_data = torch.load(model_path, map_location="cpu")
    model: torch.nn.Module = model_class(**model_data["kwargs"])
    model.load_state_dict(model_data["state_dict"])
    model.eval()
    return transform(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect glomeruli on kidney wsi")
    parser.add_argument("-v", "--verbose", help="increase logging verbosity", action="count", default=0)
    args = parser.parse_args()

    asyncio.run(main(verbosity=args.verbose))
