FROM nvcr.io/nvidia/pytorch:20.07-py3

# the above container has already conda installed with many dependencies
# install only missing dependencies
# RUN conda install -c conda-forge -y pyvips  -> pyvips doesn't work inside of container 18.04, rely on tifffile package
RUN pip install future albumentations resnest pretrainedmodels efficientnet-pytorch streamlit pylint

# set working dir
WORKDIR /app
