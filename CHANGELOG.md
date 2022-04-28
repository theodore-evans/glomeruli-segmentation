# v2.0.0
* bumped python@3.8->3.10, torch@1.8.1->1.11.0, torchvision@0.9.0->0.12.0
* rework of api interface to make use of `aiohttp`, overcoming blocking api calls and reducing job run time by 50-90%
* introduced padl as a means to streamline inference pipeline definition, replacing the `run_inference.py` module
* implemented incremental writing of patch data to mask tiles, with `overwrite`, `max` and `mean` blending models
* implemented configuration loading from .json
* various improvements to api and general app logging
* added `setup.sh` and `debug.sh` scripts to facilitate app setup and debugging with EATS
* abstracted some app logic into configuration, paving the way for a more general interface for AI solutions
# v1.1.0

* added GPU support for CUDA (tested on NVIDIA GeForce 940MX with 2GB VRAM)
* Output mask resizing now takes place on GPU using torchvision.transform, rather than openCV
* Model and input are loaded to the GPU at half precision to reduce VRAM usage with minimal impact of performance

# v1.0.0

* major refactoring to bring application into a more functional style and to align with changes to EATS 1.2.3
* added output classes, results detail posting