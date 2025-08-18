#!/bin/bash

# General Job-related configuration
export JOB_NAME="T1_tst"
# Send e-mail to address after completion. Leave as-is (blank, "") to disable.
export EMAIL="leonard.pleiss@tum.de"
# Send message to mattermost user "MATTERMOST_NAME" after completion. Leave as-is (blank, "") to disable.
export MATTERMOST_NAME=""
# Command to execute. Will be run with arguments from the
# run list file.
export EXECUTABLE="./executable"
# Resources
# How many CPUs each run requires
export CPUS_PER_RUN=1
# Specify the memory required *per cpu*. The memory requested
# per run is MEMORY_PER_CPU*CPUS_PER_RUN. Suffixes can be [K|M|G|T]
export MEMORY_PER_CPU="120G"
# Maximum time limit is 5h
export MINUTES_PER_RUN=30000
# Can be 1 or 0
export GPUS_PER_RUN=1
# Possible choices: urgent > normal
export QOS="normal"
# On which nodes to run, possible values: CPU_ONLY, GPU_ONLY, ANY, SMALL_GPU_ONLY, BIG_GPU_ONLY
export NODE_TYPE="BIG_GPU_ONLY"

# Set up your environment here, e.g., load modules, activate virtual environments
module load python/3.9.10
# module load cplex/20.1
source venv/bin/activate

# Defaults for other run-related variables.
# These can be ignored in most cases
export BASE_DIR=$(pwd)
RUN_LIST="run_list.csv"
export RESULTS_DIRECTORY="results"
export LOGS_DIRECTORY="logs"
export ERROR_LOGS_DIRECTORY="error-logs"
export SCRIPTS_DIRECTORY="scripts"
export CONSOLE_LOG_NAME="console.log"
export ERROR_CONSOLE_LOG_NAME="console-error.log"
