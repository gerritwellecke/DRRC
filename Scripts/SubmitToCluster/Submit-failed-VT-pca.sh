#!/bin/bash -l

# Use bash as shell
#$ -S /bin/bash

# Preserve environment variables
#$ -V

# Merge standard output and standard error into one file
#$ -j yes

# Standard name of the job (if none is given on the command line)
#$ -N VT_serial_pca_sg

# Logging filenames stdout & stderr
# this fails if the path does not yet exist
#$ -o /data.bmp/gwellecke/DRRC/Data/ValidTimes/1D_KuramotoSivashinsky/VT_serial_pca_sg

#$ -q teutates.q

# only failed jobs
#$ -t 1-27

# indices of failed jobs
failed_idxs=(190 253 316 380 444 508 572 636 700 764 828 892 956 1020 1148 1212 1276 1340 1404 1468 1532 1596 1660 1724 1788 1852 1915)

# Some diagnostic messages for the output
echo Started: `date`
echo on `hostname`

echo TID: ${SGE_TASK_ID} of ${SGE_TASK_LAST}

echo ------------

# restrict cores per job
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

# activate python venv
# CAVEAT: make sure all packages are installed before running this!
source /data.bmp/gwellecke/DRRC/.venv/bin/activate || exit 1

cd /data.bmp/gwellecke/DRRC

# start multiple simulations in background so the run parallel

python /data.bmp/gwellecke/DRRC/Scripts/SubmitToCluster/ClusterRun_ValidTime.py /data.bmp/gwellecke/DRRC/Scripts/SubmitToCluster/../Yml_1DKS/VT_serial_pca_sg.yml ${failed_idxs[(( ${SGE_TASK_ID} - 1 ))]}


wait

echo ------------
echo Stopped: `date`
