FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime

WORKDIR /app

RUN apt-get update -y 

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-setuptools \
    python3-wheel \
    libgl1-mesa-glx \
    libglib2.0-0

ENV PATH /root/.local/bin:/root/.poetry/bin:${PATH}

RUN mkdir -p /root/.local/bin \ 
    && ln -s $(which python3) /root/.local/bin/python \
    && curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

WORKDIR /app

COPY pyproject.toml poetry.lock /app/

RUN poetry export --without-hashes -f requirements.txt > /tmp/requirements.txt

ENV VIRTUAL_ENV /opt/venv
ENV PATH ${VIRTUAL_ENV}/bin:${PATH}
RUN python3 -m venv $VIRTUAL_ENV
RUN /opt/venv/bin/python3 -m pip install --upgrade pip
RUN pip3 install -r /tmp/requirements.txt

COPY ./src /app

# Download model weights
RUN mkdir -p /model && curl -OJ https://nextcloud.empaia.org/s/dMmHbbAEfwccnRg/download && mv glomeruli_segmentation_16934_best_metric_full.model-384e1332.pth /model
ENV MODEL_PATH /model/glomeruli_segmentation_16934_best_metric_full.model-384e1332.pth
      
CMD ["python3", "main.py", "-v"]
