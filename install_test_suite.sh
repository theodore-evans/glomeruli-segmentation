#!/usr/bin/env bash

if [[ $# != 2 || $(basename -- $2) != "empaia-app-test-suite" ]]; then
    echo "Usage: ./install_test_suite.sh <path-to-empaia-app-test-suite>"
fi

docker_container_name=$1
test_suite_dir=$2
test_suite_path_in_container="/test_suite"

docker cp $test_suite_dir $docker_container_name:$test_suite_path_in_container
docker exec -u 0 -w $test_suite_path_in_container $docker_container_name /bin/bash -c \
    "./empaia_app_test_suite_services/run_wheel_build.sh; \
    ./run_services_deployment.sh; \
    pip install .;"