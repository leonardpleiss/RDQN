#!/bin/bash

# This script is executed when a run fails

# The exit code of your program
STATUS_CODE=$1
# The failed run's name
RUN_NAME=$2
shift 2
# Any arguments passed to the specified executable
ARGS="$@"

# One could e.g. send an email on failure:
#
# if [ -n ${EMAIL} ]; then
#         cat <<-EOF | mail ${EMAIL} -s "[CLUSTER] Error occured in ${JOB_NAME}!" -a "From:$(whoami)@master.osm.wi.tum.de"
#         Run ${RUN_NAME} executed with "$ARGS" exited with code ${STATUS_CODE}!
# EOF
#fi
