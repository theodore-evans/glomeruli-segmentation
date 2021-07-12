from math import gamma
from app.mock_api import MockAPI as API
from app.inference_runner import InferenceRunner

from data.preprocessing import raw_test_transform

from cem import *

def main():
    api = API()
    inference = InferenceRunner(
        "data_empaia/hacking_kidney_16934_best_metric.model-384e1332.pth", data_transform=raw_test_transform()
    )

    # tile fetcher for defined rectangle, write cem tile fetcher
    kidney_wsi = api.get_input("kidney_wsi")

    def tile_request(rect):
        return api.get_wsi_tile(kidney_wsi, rect)

    tile_fetcher = CEMTileFetcher(tile_request, api.get_input("my_rectangle"))
    quantizer = CEMQuantizer(sum)
    infer_wrapper = CEMInferenceWrapper(inference, tile_fetcher, quantizer)

    cem = CEM(
        mode="PP",
        model_infer=infer_wrapper,
        AE=None,
        org_img=tile_fetcher._orig_region,
        c=1,
        beta=1e-1,
        l1=0.0001,
        l2=0.0001,
        gamma=0,
        K=2.0,
        mu=None,
        max_iter=50,
    )

    res = cem.run()


if __name__ == "__main__":
    main()
