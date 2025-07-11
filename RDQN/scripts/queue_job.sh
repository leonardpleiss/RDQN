#!/bin/bash

export TARGET_DIR_NAME=$1

# Set up directories for this run

export TARGET_LOG_DIRECTORY="${BASE_DIR}/${LOGS_DIRECTORY}/${TARGET_DIR_NAME}"
export TARGET_RESULTS_DIRECTORY="${BASE_DIR}/${RESULTS_DIRECTORY}/${TARGET_DIR_NAME}"
export TARGET_ERROR_LOG_DIRECTORY="${BASE_DIR}/${ERROR_LOGS_DIRECTORY}/${TARGET_DIR_NAME}"

for dir in "${TARGET_LOG_DIRECTORY}" "${TARGET_RESULTS_DIRECTORY}" "${TARGET_ERROR_LOG_DIRECTORY}"; do
    if ! [ -e "${dir}" ]; then
        mkdir -p "${dir}"
    fi
done

export TARGET_STDOUT_LOG="${TARGET_LOG_DIRECTORY}/${CONSOLE_LOG_NAME}"
export TARGET_STDERR_LOG="${TARGET_LOG_DIRECTORY}/${ERROR_CONSOLE_LOG_NAME}"

case ${NODE_TYPE} in 
    "GPU_ONLY")
        _NODES="--exclude=osm-cpu-[1-6]"
        ;;
    "CPU_ONLY")
        _NODES="--exclude=osm-gpu-[1-5]"
        ;;
    "SMALL_GPU_ONLY")
        _NODES="--exclude=osm-gpu-[3-5],osm-cpu-[1-6]"
        ;;
    "BIG_GPU_ONLY")
        _NODES="--exclude=osm-gpu-[1-2],osm-cpu-[1-6]"
        ;;
    "ANY")
        _NODES=""
        ;;
    *)
        echo "Error, unknown node type: ${NODE_TYPE}"
        ;;
esac

srun -n 1 --job-name="${TARGET_DIR_NAME}" --cpus-per-task=${CPUS_PER_RUN} --gpus-per-task=${GPUS_PER_RUN} --mem-per-cpu=${MEMORY_PER_CPU} --time=${MINUTES_PER_RUN} --qos="${QOS}" ${_NODES} --export=ALL --output="${TARGET_STDOUT_LOG}" --error="${TARGET_STDERR_LOG}" "${BASE_DIR}/${SCRIPTS_DIRECTORY}/dispatch_instance.sh" "$@"
slurm_status_code=$?

if [ $slurm_status_code -ne 0 ]; then
	cp -a "${TARGET_LOG_DIRECTORY}/." "${TARGET_ERROR_LOG_DIRECTORY}/"
    ${BASE_DIR}/${SCRIPTS_DIRECTORY}/on_error.sh ${slurm_status_code} "${TARGET_DIR_NAME}" "$@"
fi

exit ${slurm_status_code}
