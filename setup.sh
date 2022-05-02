#!/bin/bash

# setup stage
inputs_dir=inputs

# if eats command is not recognised, abort
if ! command -v eats >/dev/null 2>&1; then
    echo "eats command not found. Aborting."
    exit 1
fi

# if app.env file is missing, abort
if [ ! -f app.env ]; then
    echo "app.env file not found. Aborting." >&2
    exit 1
fi

export $(xargs < app.env)

# if APP_ID is missing, abort
if [ -z "$APP_ID" ]; then
    echo "APP_ID is missing. Aborting" >&2
    exit 1
fi

# if registering a job fails, abort
if [ ! -z "$(eats jobs register $APP_ID $inputs_dir --v1 2>&1 > job.env | tee /dev/tty |  grep -i 'error\|exception')" ]; then
    echo "failed to register job for APP_ID='$APP_ID'. Aborting" >&2
    exit 1
fi

export $(xargs < job.env)
eats jobs set-running $EMPAIA_JOB_ID