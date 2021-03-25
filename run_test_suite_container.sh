#!/usr/bin/env bash

app_dir=$(pwd)
data_dir="/mnt/hxstorage1/evans/data/hubmap-kidney-segmentation/test"

docker run -it --rm \
 -v /var/run/docker.sock:/var/run/docker.sock \
 -v $app_dir:$app_dir \
 --net=host \
 --name $USER-empaia-test-suite \
 empaia-test-suite \
 /bin/bash
