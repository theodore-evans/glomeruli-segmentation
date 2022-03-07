# v1.1.0

* added GPU support for CUDA (tested on NVIDIA GeForce 940MX with 2GB VRAM)
* Output mask resizing now takes place on GPU using torchvision.transform, rather than openCV
* Model and input are loaded to the GPU at half precision to reduce VRAM usage with minimal impact of performance

# v1.0.0

* major refactoring to bring application into a more functional style and to align with changes to EATS 1.2.3
* added output classes, results detail posting