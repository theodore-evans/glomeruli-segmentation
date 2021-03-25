#!/usr/bin/env bash

app_dir=$(pwd)
if [ $# -eq 1 ]; then app_dir=$1; fi

echo "App directory: $app_dir"

empaia-app-test-suite \
 $app_dir/kidney_segmentation_app.json \
 $app_dir/inputs \
 $app_dir/outputs \
 evans-hacking-kidney