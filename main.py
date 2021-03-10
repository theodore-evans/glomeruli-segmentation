import argparse

from data.wsi_data_loader import WSITileDataset
from app.inference_runner import InferenceRunner
from app.postprocessing import EntityExtractor
from api import get_input, get_wsi_tile, post_output, put_finalize

from data.data_processing import kaggle_test_transform

def run_app():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, type=str, help='path to a pre-trained model (.pth)')
    parser.add_argument('--window-size', default=1024, type=int, help='')
    args = parser.parse_args()

    model_path = args.model

    my_wsi = get_input("my_wsi")
    my_rectangle = get_input("my_rectangle")
    wsi_tile = get_wsi_tile(my_wsi, my_rectangle)

    tiled_dataset = WSITileDataset(wsi_tile=wsi_tile, )
    inference_runner = InferenceRunner(model_path=model_path, data_transform=kaggle_test_transform())
    entity_extractor = EntityExtractor() # placeholder mockup, doesn't do anything

    segmentation_mask = inference_runner(tiled_dataset)
    glomerulus_contours = entity_extractor.extract_contours(segmentation_mask)
    glomerulus_count = entity_extractor.count_entities(glomerulus_contours)

    glomerulus_count = {
        "name": "entity count glomeruli",
        "value": glomerulus_count
    }

    post_output("glomerulus_count", glomerulus_count)
    put_finalize()
    
if __name__ == '__main__':
    run_app()