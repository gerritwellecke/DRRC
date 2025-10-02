Running a job on the cluster
============================

Installing python on the cluster
--------------------------------

At MPI-DS
^^^^^^^^^
To have a current version of python I recommend installing `miniconda <https://docs.conda.io/projects/miniconda/en/latest/>`_.
Download the Linux 64-bit installer and use :code:`scp` to put it in your home directory on the cluster.
Then ssh into the cluster and use the "Quick command line install" for Linux from the link above.

At GWDG 
^^^^^^^
They have decent python.
Run :code:`module load python` when setting up the virtual environment for the first time.
Thereafter simply activating the virtual environment is enough.

Submission Scripts
------------------
A cluster job can be started using :code:`SubmitToCluster.sh`.
This script expects a YAML configuration file and a specification of the type of cluster run you want (see help for options). 

Under the hood :code:`ClusterRun_<type>.py` takes care of initial setup, i.e. ensuring that data output directories exist and that the submission script is rendered according to the YAML configuration.

Under the hood :func:`drrc.Config.generate_submission_script_from_YAML` parses the YAML configuration and fills a template for a specified queueing system.
