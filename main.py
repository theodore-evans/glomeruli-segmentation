import argparse

from data.wsi_tile_fetcher import WSITileFetcher
from app.inference_runner import InferenceRunner
from app.api import get_wsi_data, get_wsi_tile, post_output, put_finalize
from app.data_types import TileRequest, WSI
from app.entity_extractor import EntityExtractor
from data.preprocessing import kaggle_test_transform
    
def run_app():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsi_id', default=None, type=str, help='wsi_id for wsi to fetch from EMPAIA API')
    parser.add_argument('--model', default=None, type=str, help='path to a pre-trained model (.pth)')
    parser.add_argument('--window_size', default=1024, type=int, help='')
    args = parser.parse_args()

    model_path: str = args.model
    wsi_id: str = args.wsi_id
    window_size: int = args.window_size

    wsi_data: WSI = get_wsi_data(wsi_id)
    wsi_height: int = wsi_data['extent']['y']
    wsi_width: int = wsi_data['extent']['x']

    tile_request: TileRequest = lambda rect: get_wsi_tile(wsi_data, rect)
    wsi_tile_fetcher = WSITileFetcher(
        tile_request=tile_request,
        window_size=window_size,
        original_size=(wsi_height, wsi_width)
        )
    
    inference_runner = InferenceRunner(model_path=model_path, data_transform=kaggle_test_transform())
    entity_extractor = EntityExtractor() #TODO: Implement. placeholder mockup, doesn't do anything yet

    segmentation_mask = inference_runner(wsi_tile_fetcher)
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