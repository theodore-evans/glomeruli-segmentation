import os
import requests
from PIL import Image

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

def post_output(key: str, data: dict) -> dict:
    """
    post output data by key as defined in EAD
    """
    url = f"{APP_API}/v1/{JOB_ID}/outputs/{key}"
    r = requests.post(url, json=data, headers=HEADERS)
    r.raise_for_status()
    return r.json()

def get_wsi_tile(my_wsi: dict, my_rectangle: dict) -> Image.Image:
    """
    get a WSI tile on level 0

    Parameters:
        my_wsi: contains WSI id (and meta data)
        my_rectangle: tile position on level 0
    """
    x, y = my_rectangle["upper_left"]
    width = my_rectangle["width"]
    height = my_rectangle["height"]

    wsi_id = my_wsi["id"]
    level = 0
    
    tile_url = f"{APP_API}/v1/{JOB_ID}/regions/{wsi_id}/level/{level}/start/{x}/{y}/size/{width}/{height}"
    # not matching documentation at https://gitlab.cc-asp.fraunhofer.de/empaia/platform/data/medical-data-service

    r = requests.get(tile_url, headers=HEADERS)
    r.raise_for_status()

    return Image.open(r.content)
    
my_wsi = get_input("my_kidney_wsi")
my_rectangle = get_input("my_rectangle")

