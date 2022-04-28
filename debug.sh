#!/bin/bash

# ensure setup.sh exists
if [ ! -f setup.sh ]; then
    echo "setup.sh not found. Aborting."
    exit 1
fi

# run ./setup.sh and abort if it fails
if [ ! -z "$(./setup.sh 2>&1 | tee /dev/tty |  grep -i 'abort')" ]; then
    echo "failed to run setup.sh. Aborting"
    exit 1
fi

# run stage
export $(xargs < job.env)
echo "EMPAIA_JOB_ID="$EMPAIA_JOB_ID
export $(echo "EMPAIA_APP_API=http://localhost:8888/app-api" | tee /dev/tty)
export $(echo "MODEL_PATH=glomeruli_segmentation_16934_best_metric.model-384e1332.pth" | tee /dev/tty)
export $(echo "CONFIG_PATH=configuration.json" | tee /dev/tty)

echo "Debug process ready, listening on localhost:5678"
python -m debugpy --listen 5678 --wait-for-client -m glomeruli_segmentation $@