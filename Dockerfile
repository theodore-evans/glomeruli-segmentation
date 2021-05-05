FROM nvcr.io/nvidia/pytorch:21.02-py3

WORKDIR /app

COPY . /app

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

ENV PATH /root/.local/bin:/root/.poetry/bin:${PATH}

RUN poetry install

RUN mkdir -p /app/model && curl -OJ https://nx9836.your-storageshare.de/s/HSq8StKLB6WYncy/download && mv hacking_kidney_16934_best_metric.model-384e1332.pth /app/model

ENTRYPOINT poetry run python3 /app/main.py
