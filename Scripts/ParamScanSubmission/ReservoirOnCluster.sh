#!/bin/bash

usage () {
        cat <<HELP_USAGE
        Submit script to cluster.

        Usage:
        $(basename $0) [PathToYaml]

        Args:
            1 PathToYaml: Path to .yml file (can be relative path!)

HELP_USAGE
}

# Argument checking
if [[ $# != 1 ]]
    then
        usage
        echo "No arguments provided. Aborting."
    exit
fi

gitroot=$(git rev-parse --show-toplevel)

# Activate environment
source $gitroot/.venv/bin/activate || exit 1


# Create submission script & output directories
JOBSCRIPT_PATH=$(python "${gitroot}/Scripts/ParamScanSubmission/ClusterRun.py" $1 0)

# Submit reservoir computer job to queue
qsub $JOBSCRIPT_PATH
