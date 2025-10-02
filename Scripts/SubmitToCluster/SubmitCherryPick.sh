#!/bin/bash

usage () {
        cat <<HELP_USAGE
        Submit script to cluster.

        Usage:
        $(basename $0) [PathToYaml] [submission command]

        Args:
            1 PathToYaml: Path to .yml file (can be relative path!)
            2 submission command: either 'qsub' or 'sbatch'

HELP_USAGE
}

# Argument checking
if [[ $# != 2 ]]
    then
        usage
    exit 1
fi

gitroot=$(git rev-parse --show-toplevel)

# Activate environment
source $gitroot/.venv/bin/activate || exit 1

# Create submission script & output directories
SUBMIT_JOBSCRIPT=$(python "${gitroot}/Scripts/SubmitToCluster/CherryPickRun.py" $1 0)

# Submit reservoir computer job to queue
$SUBMIT_JOBSCRIPT
