#!/bin/bash

pid=$2
start_time=$8
duration=${10}

module purge

source {VENV_PATH}

pid_dir={OUTPUT_DIR}/logs/${pid}
#Create a pid directory if it doesn't exist
mkdir -p ${pid_dir}

python {OUTPUT_DIR}/computation.py "$@" > ${pid_dir}/${start_time}_${duration}.log 2>&1
