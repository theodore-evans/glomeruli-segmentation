#!/usr/bin/env sh

docker_image_tag=$USER-hacking-kidney

if [ "$#" -eq  "1" ]; then
    docker_image_tag=$1
fi

docker build -t $docker_image_tag .