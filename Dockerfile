# build stage
FROM docker.io/ubuntu:20.04 AS builder

RUN apt-get update \
&& apt-get install -y python3-venv python3-pip git curl

# build module
ENV PATH /root/.local/bin:/root/.poetry/bin:${PATH}
RUN mkdir -p /root/.local/bin \
&& ln -s $(which python3) /root/.local/bin/python \
&& curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
COPY . /root/app
WORKDIR /root/app
RUN poetry export --without-hashes -f requirements.txt > requirements.txt
RUN poetry build

# install stage
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN apt-get update -y 

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0

RUN : \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        python3.10-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && :

RUN python3.10 --version

ENV VIRTUAL_ENV /opt/app/venv
ENV PATH ${VIRTUAL_ENV}/bin:${PATH}
RUN python3.10 -m venv $VIRTUAL_ENV \
    && /opt/app/venv/bin/python3 -m pip install --upgrade pip \
    && pip3 install wheel
COPY --from=builder /root/app/requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt
COPY --from=builder /root/app/dist/*.whl /tmp
RUN pip3 install /tmp/*.whl

# Download model weights
RUN mkdir -p /model && curl -OJ https://nx9836.your-storageshare.de/s/gaqByayBJXcBmPQ/download && mv glomeruli_segmentation_16934_best_metric.model-384e1332.pth /model
ENV MODEL_PATH /model/glomeruli_segmentation_16934_best_metric.model-384e1332.pth

# Copy configuration file
COPY configuration.json /tmp
ENV CONFIG_PATH /tmp/configuration.json
      
CMD ["python3.10", "-m", "glomeruli_segmentation", "-v"]
