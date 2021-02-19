#!/usr/bin/env sh

docker run -it --rm \
  --gpus all \
  --shm-size=32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 8501:8501 \
  -v `pwd`:/workspace/hacking_kidney \
  -v /data/hubmap-kidney-segmentation:/data/hubmap-kidney-segmentation \
  hacking_kidney:latest streamlit run demo.py -- --image-size=1024 --mode=valid --model hacking_kidney_16934_best_metric.model-384e1332.pth