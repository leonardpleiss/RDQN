#!/bin/bash

function gather_and_setup () {
    for dir in $(ls "${BASE_DIR}"); do
	case "${dir}" in
		"${RESULTS_DIRECTORY}"|"${ERROR_LOGS_DIRECTORY}"|"${LOGS_DIRECTORY}") ;;
		*)
			ln -s "${BASE_DIR}/${dir}"
	esac
    done

    # Create local directory for output files
    mkdir -p "${RESULTS_DIRECTORY}"
    mkdir -p "${ERROR_LOGS_DIRECTORY}"
    mkdir -p "${LOGS_DIRECTORY}"
}

RUN_NAME=$1
shift

exit_code=0

# Create a unique temporary directory on the local filesystem
RUN_DIRECTORY=$(mktemp -d -p /var/scratch "job-template-XXXXXXXXX")

if [ $? -ne 0 ]; then
    echo "[Template]: Error, could not create temp dir with prefix (${RUN_NAME}) in /var/scratch" && exit 1
fi

cd ${RUN_DIRECTORY}

# Set up 
gather_and_setup
exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo "[Template]: Error, directory setup failed: $exit_code" && exit $exit_code
fi

"${BASE_DIR}/${SCRIPTS_DIRECTORY}/prepare.sh"

exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "[Template]: Error running \"${BASE_DIR}/${SCRIPTS_DIRECTORY}/prepare.sh\": $exit_code" && exit $exit_code
fi

# Write SLURM info to logs/
echo "$SLURM_JOB_ID" > "${TARGET_LOG_DIRECTORY}/slurm_job_id.txt"
echo "$SLURM_JOB_NAME" > "${TARGET_LOG_DIRECTORY}/slurm_job_name.txt"

echo "--------- Start of ${EXECUTABLE} ---------"
# Run the command
${EXECUTABLE} "$@"
executable_exit_code=$?
echo "--------- End of ${EXECUTABLE} ---------"

"${BASE_DIR}/${SCRIPTS_DIRECTORY}/epilog.sh"

exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "[Template]: Error running \"${BASE_DIR}/${SCRIPTS_DIRECTORY}/epilog.sh\": $exit_code" && $exit_code
fi

# Copy output files back to base directory/shared folder

if [ $executable_exit_code -ne 0 ] ; then
    echo "[Template]: Error running \"${EXECUTABLE} $@\": $executable_exit_code. See ${TARGET_ERROR_LOG_DIRECTORY}"
fi

cp -a logs/* "${TARGET_LOG_DIRECTORY}/"
cp -a results/* "${TARGET_RESULTS_DIRECTORY}/"

# Clean up
cd /tmp
rm -rf ${RUN_DIRECTORY}

exit ${executable_exit_code}
