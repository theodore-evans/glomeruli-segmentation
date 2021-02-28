#!/usr/bin/env sh

if [$# -ne 3]; then echo "Arguments expected: data path, EMPAIA test suite path, model path"
else
    data_path=$1
    test_suite_path=$2
    model=$3
fi

echo "$data_path"
echo "$test_suite_path"
echo "$model"

docker run -it --rm \
  --gpus all \
  --shm-size=32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 8501:8501 \
  -v $(pwd):/app \
  -v $data_path:/data/hubmap-kidney-segmentation \
  -v $test_suite_path:/test-suite \
  $USER-hacking-kidney:latest \
  streamlit run demo.py -- --image-size=1024 --mode=valid --model $model
