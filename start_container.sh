#!/usr/bin/env bash

set -e

# Defaults
data_dir="$(pwd)/data/hubmap-kidney-segmentation"
model_path="$(pwd)/hacking_kidney_16934_best_metric.model-384e1332.pth"
image_size=1024
docker_image_tag=$USER-hacking-kidney
docker_container_name=$docker_image_tag
empaia_app_api="http://localhost:80"
empaia_job_id="someId"
empaia_token="someToken"
#TODO: add argument parsing for environment variables, or package this all up into a docker-compose configuration

print_usage () {
  echo -e "usage: $0 [-d data_path] [-m model_path] [-t test_suite_dir] [entry_point]\n"
}

print_example () {
  echo -e "example: $0 -d /mnt/data/hubmap-kidney-segmentation -m \$(pwd)/models/hacking_kidney.pth - t ~/empaia-test-suite /bin/bash\n"
}

print_help () {
  print_usage
  echo -e "OPTION:"
  echo -e "-d   data_path:  Absolute path to directory containing HuBMAP: Hacking Kidney data" 
  echo -e "     (default: \$(pwd)/data/hubmap-kidney-segmentation)"
  echo -e "-m   model_path: Absolute path to trained inference model in .pth format"
  echo -e "     (default: \$(pwd)/hacking_kidney_16934_best_metric.model-384e1332.pth)"
  echo -e "-t   test_suite_dir: Path to empaia-app-test-suite"
  echo -e "     (default: none, test suite will not be mounted)"
  echo -e "-i   docker_image_tag: name of Docker image for app"
  echo -e "     (default: \$USER-hacking-kidney)"
  echo -e "entry_point: command to run when the container starts"
  echo -e "     (default: streamlit run demo.py -- --image-size=1024 --mode=valid --model <trained_model>)\n"
  print_example
}

check_for_docker_image () {
  if [[ -z $(docker images -q $docker_image_tag) ]]; then
    read -p "Docker image $1 not found. Build now? (y/n): " yn
    case $yn in
      [Yy]* )
          bash build_container.sh $docker_image_tag;;
      * )
          exit;;
    esac
  fi
}

check_for_model () {
  if [[ ! -z $model_path ]]; then
    if [[ -f "$model_path" && "${model_path##*.}" == "pth" ]]; then
      model_filename="${model_path##*/}"
      model_path_in_container="/model/$model_filename"
      mount_model_file="--mount type=bind,source=$model_path,target=$model_path_in_container"
    else
      echo -e "Model file $model_path not found or wrong extension. \nUse -d <PATH> to specify absolute path to a .pth model "
      exit
    fi
  fi
}

check_for_entry_point () {
  if [[ $entry_point == "demo" ]]; then 
    entry_point=$(run_demo $image_size $model_path_in_container)
  fi
}

run_demo () {
  echo "streamlit run demo.py -- --image-size=$1 --mode=valid --model $2"
}

# Script starts here
while getopts d:m:t:i:h flag
  do
      case "${flag}" in
          d) data_dir=${OPTARG};;
          m) model_path=${OPTARG};;
          t) test_suite_dir=${OPTARG};;
          i) docker_image_tag=${OPTARG};;
          h) print_help; exit;;
      esac
  done
shift $((OPTIND-1))
entry_point=$1

check_for_model
check_for_entry_point
check_for_docker_image

echo
echo "Running Docker image $docker_image_tag with entry point $entry_point"
echo
echo "Data path: $data_dir"
echo "Model path: $model_path" 
if [[ ! -z mount_test_suite ]]; then echo "Test suite path: $test_suite_dir"; fi

docker run -d \
  --gpus all \
  --shm-size=32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 8501:8501 \
  --mount type=bind,source=$(pwd),target=/app \
  --mount type=bind,source=$data_dir,target=/data/hubmap-kidney-segmentation \
  $mount_model_file \
  -e EMPAIA_APP_API=$empaia_app_api \
  -e EMPAIA_JOB_ID=$empaia_job_id \
  -e EMPAIA_TOKEN=$empaia_token \
  --name $docker_container_name \
  $docker_image_tag \
  $entry_point

if [[ ! -z "$test_suite_dir" ]]; then
    bash install_test_suite.sh $docker_container_name $test_suite_dir
fi

docker exec -it $docker_container_name /bin/bash