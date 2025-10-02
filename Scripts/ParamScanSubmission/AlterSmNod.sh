#!/bin/bash

# take two inputs, the current job_id and the previous one
usage () {
        cat <<HELP_USAGE
        Alter submitted jobs for concurrency and causality.

        Usage:
        $(basename $0) [job_id] [previous job_id]

        Args:
            1 job_id: id of the just submitted job
            2 prev. job_id: id of the preceding job

HELP_USAGE
}

# Argument checking
if [[ $# != 2 ]]
    then
        usage
        echo "No arguments provided. Aborting."
    exit
fi

# hold current job
qhold $1

# alter the job
qalter -tc 47 -p -1 -hold_jid $2

# release current job
qrls $1
