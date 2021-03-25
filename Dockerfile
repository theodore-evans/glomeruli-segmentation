FROM nvcr.io/nvidia/pytorch:21.02-py3

# RUN conda install -c conda-forge -y pyvips  -> pyvips doesn't work inside of container 18.04, rely on tifffile package
COPY . /app
WORKDIR /app
COPY environment.yml /app/environment.yml
RUN conda config --add channels conda-forge \
    && conda env create -n hacking-kidney -f environment.yml \
    && rm -rf /opt/conda/pkgs/*

ENV PATH /opt/conda/envs/hacking-kidney/bin:$PATH

ENTRYPOINT python3 -u /app/main.py