#!/usr/bin/env sh

docker_image_tag=$USER-hacking-kidney

if [ "$#" -eq  "0" ]; then
    docker_image_tag=$1
fi

docker build --tag $docker_image_tag --build-arg current_user=$USER .