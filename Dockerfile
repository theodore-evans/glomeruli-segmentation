# build stage
FROM docker.io/ubuntu:20.04 AS builder
RUN apt-get update \
&& apt-get install -y python3-venv python3-pip git curl

# service name
RUN apt-get install -y git curl
ENV PATH /root/.local/bin:/root/.poetry/bin:${PATH}
RUN mkdir -p /root/.local/bin \
&& ln -s $(which python3) /root/.local/bin/python \
&& curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
COPY . /root/app
WORKDIR /root/app
RUN poetry export --without-hashes -f requirements.txt > requirements.txt
RUN poetry build

# install stage
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

RUN apt-get update -y 

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git \
    curl \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0

RUN echo python3 --version

ENV VIRTUAL_ENV /opt/app/venv
ENV PATH ${VIRTUAL_ENV}/bin:${PATH}
RUN python3 -m venv $VIRTUAL_ENV \
&& /opt/app/venv/bin/python3 -m pip install --upgrade pip \
&& pip3 install wheel
COPY --from=builder /root/app/requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt
COPY --from=builder /root/app/dist/*.whl /tmp
RUN pip3 install /tmp/*.whl

# Download model weights
RUN mkdir -p /model && curl -OJ https://nx9836.your-storageshare.de/s/7FGbJRn97NYB5pD/download && mv glomeruli_segmentation_16934_best_metric.model-384e1332.pth /model
ENV MODEL_PATH /model/glomeruli_segmentation_16934_best_metric.model-384e1332.pth
      
CMD ["python3", "-m", "glomeruli_segmentation", "-v"]
