#!/usr/bin/env sh

# Defaults
default_data_path="/mnt/hxstorage1/evans/data/hubmap-kidney-segmentation"
default_model_path="hacking_kidney_16934_best_metric.model-384e1332.pth"
default_test_suite_path="../empaia-test-suite"

if [ $# -eq 0 ]; then
    data_path=default_data_path
    test_suite_path=default_test_suite_path
    model=default_model_path
else
    if [$# -ne 3]; then echo "Arguments expected: data path, EMPAIA test suite path, model path"
    else
        data_path=$1
        test_suite_path=$2
        model=$3
    fi
fi

docker run -it --rm \
  --gpus all \
  --shm-size=32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 8501:8501 \
  -v $pwd/workspace/hacking_kidney \
  -v $data_path:/data/hubmap-kidney-segmentation \
  -v $test_suite_path:/workspace/test-suite \
  hacking_kidney:latest streamlit run demo.py -- --image-size=1024 --mode=valid --model $model