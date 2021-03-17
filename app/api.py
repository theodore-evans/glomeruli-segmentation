from app.schema import Schema
import os
import io
import requests
from PIL import Image

from app.data_types import Rectangle, WSI

APP_API = os.environ["EMPAIA_APP_API"]
JOB_ID = os.environ["EMPAIA_JOB_ID"]
TOKEN = os.environ["EMPAIA_TOKEN"]
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
        
def put_finalize():
    """
    finalize job, such that no more data can be added and to inform EMPAIA infrastructure about job state
    """
    url = f"{APP_API}/v1/{JOB_ID}/finalize"
    r = requests.put(url, headers=HEADERS)
    r.raise_for_status()

def get_input(key: str) -> dict:
    """
    get input data by key as defined in EAD
    """
    url = f"{APP_API}/v1/{JOB_ID}/inputs/{key}"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return r.json()

def get_wsi_data(wsi_id: str) -> WSI:
    response = get_input(wsi_id)
    Schema(WSI).validate(response)
    wsi_data: WSI = { 
        "id" : response["id"], 
        "extent" : response["extent"],
        "num_levels" : response["num_levels"],
        "pixel_size_nm" : response["num_levels"],
        "tile_extent": response["tile_extent"],
        "levels": response["levels"]
        }
    return wsi_data

def post_output(key: str, data: dict) -> dict:
    """
    post output data by key as defined in EAD
    """
    url = f"{APP_API}/v1/{JOB_ID}/outputs/{key}"
    r = requests.post(url, json=data, headers=HEADERS)
    r.raise_for_status()
    return r.json()

def get_wsi_tile(wsi: WSI, rectangle_to_fetch: Rectangle) -> Image.Image:
    """
    get a WSI tile on level 0

    Parameters:
        my_wsi: contains WSI id (and meta data)
        my_rectangle: tile position on level 0
    """
    x, y = rectangle_to_fetch["upper_left"]
    width = rectangle_to_fetch["width"]
    height = rectangle_to_fetch["height"]
    level = rectangle_to_fetch["level"]
    
    wsi_id = wsi["id"]
 
    tile_url = f"{APP_API}/v1/{JOB_ID}/regions/{wsi_id}/level/{level}/start/{x}/{y}/size/{width}/{height}"

    r = requests.get(tile_url, headers=HEADERS)
    r.raise_for_status()

    return Image.open(io.BytesIO(r.content))

