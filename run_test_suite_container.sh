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

# Comments on arguments:
#  -v /var/run/docker.sock:/var/run/docker.sock
#       Allows the test suite to access the Docker context of the host machine in order to find and start containers
#       NOTE: docker.io must also be installed on the test suite container (see Dockerfile)
#  -v $app_dir:$app_dir 
#       The app data (EAD, input & output directories) must be made available within the test suite container 
#       at the same path as on the host, as the test suite will check that these paths are valid (within the container) 
#       before mounting them to the test suite services container (from the host machine)
#  --net=host
#       The test suite expects the services to be made available at localhost:<exposed port for service>, this line allows
#       the container to share the host's networking namespace such that these requests can be received
#       i.e. the container network will not be allocated a separate IP-address, as it would with this line