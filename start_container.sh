#!/usr/bin/env sh

usage="Usage: $0 [-d data_path] [-m model_path] [-t test_suite_path]\n \
  -d data_path: Relative path to directory containing HuBMAP: Hacking Kidney data\n\t \
  \t(default: ./data/hubmap-kidney-segmentation)\n\t \
  -m model_path: Relative path to inference model\n\t \
  \t(default: ./hacking_kidney_16934_best_metric.model-384e1332.pth)\n\t \
  -t test_suite_path: Path to empaia-test-suite\n\t
  \t(default: none, test suite will not be mounted)"

data_path="/data/hubmap-kidney-segmentation"
model="hacking_kidney_16934_best_metric.model-384e1332.pth"

while getopts d:m:t:h flag
do
    case "${flag}" in
        d) data_path=${OPTARG};;
        m) model=${OPTARG};;
        t) test_suite_path=${OPTARG};;
        h) echo $usage; exit;
    esac
done

echo "Data: $data_path"
echo "Model: $model"

if [ ! -z "$test_suite_path" ]; then
  echo "Test suite: $test_suite_path"
  mount_test_suite="-v $test_suite_path:/test-suite"
fi

docker run -it --rm \
  --gpus all \
  --shm-size=32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 8501:8501 \
  -v $(pwd):/app \
  -v $data_path:/data/hubmap-kidney-segmentation \
  $mount_test_suite \
  $USER-hacking-kidney:latest \
  /bin/bash
  #streamlit run demo.py -- --image-size=1024 --mode=valid --model $model
