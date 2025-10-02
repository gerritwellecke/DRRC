"""
Wrapper class for YAML configurations.

Use logger.INFO_1RUN for verbose output of this module's functionality.

.. codeauthor:: Gerrit Wellecke
"""

import itertools
import logging as log
import sys
import warnings
from pathlib import Path
from shutil import copy

import h5py
import numpy as np
import yaml
from git import Repo
from more_itertools import divide

from drrc.tools.logger_config import drrc_logger as logger

CLUSTER_CORES = 32

# get logger levels
loglevel_info_single_run = log.getLevelName("INFO_1RUN")
loglevel_info_multiple_run = log.getLevelName("INFO_nRUN")


class IndentDumper(yaml.Dumper):
    """YAML dumper needed for correct printing of config"""

    def increase_indent(self, flow=False, indentless=False):
        return super(IndentDumper, self).increase_indent(flow, indentless)


class Config:
    """Configuration of a system from a YAML file"""

    def __init__(self, proj_path: Path):
        """Initializes the config from a path

        Args:
            proj_path (str): absolute path from the git repository's root
        """
        self.path = Path(proj_path)
        self.dict = self.parse_YAML()
        self.name = self.path.stem
        self.max_length = self["Jobscript"]["max_job_count"]

    @property
    def path(self):
        """pathlib.Path: Absolute path to the configuration YAML file for the current
        system."""
        return self._path

    @path.setter
    def path(self, proj_path: Path):
        """prepend git root"""
        self._path = self.get_git_root() / proj_path.absolute()

    @property
    def max_length(self):
        """int: Maximum length of param_scan_list"""
        return self._max_length

    @max_length.setter
    def max_length(self, min_job_count):
        """Check if each job runs multiple parallel tasks"""
        try:
            self._max_length = min_job_count * self["Jobscript"]["tasks_per_job"]
        except:
            self._max_length = min_job_count

    @staticmethod
    def get_git_root() -> Path:
        """Get root of the current git repository

        Returns:
            :class:`pathlib.Path`: absolute path to the git-root for the current system

        Warning:
            This does not really belong in this class.
        """
        path = Path(
            Repo(".", search_parent_directories=True).git.rev_parse("--show-toplevel")
        ).absolute()
        return path

    def get_data_dir(self) -> Path:
        """Get absolute path to the current systems data directory.

        Returns:
            :class:`pathlib.Path`: absolute path to the data directory for the current system
        """
        return self.get_git_root() / Path(
            f"Data/{self['Data']['model_dimension']}D_{self['Data']['model_name']}/"
        )

    def load_training_dataset(self, index: int) -> np.ndarray:
        """Load and return the training data.

        Args:
            index: The index of the training data set to load.

        Returns:
            One training data set with minimal temporal length needed for iterative timeseries predictions, i.e. :code:`temporal_length=(transient_steps + training_steps + 1)`.

        Warning:
            This function should be modified to load only the amount of variables needed, to save memory.
        """

        logger.log(loglevel_info_single_run, "Loading training data.")
        transient_steps = int(
            self["Data"]["Usage"]["tansient_length"]
            // self["Data"]["Creation"]["TemporalParameter"]["dt"]
        )
        training_steps = int(
            self["Data"]["Usage"]["training_length"]
            // self["Data"]["Creation"]["TemporalParameter"]["dt"]
        )

        # Check if the number of training steps is larger than the length of the training data
        if transient_steps + training_steps + 1 > (
            self["Data"]["Creation"]["TemporalParameter"]["training_length"]
            // self["Data"]["Creation"]["TemporalParameter"]["dt"]
        ):
            training_steps = (
                (
                    self["Data"]["Creation"]["TemporalParameter"]["training_length"]
                    // self["Data"]["Creation"]["TemporalParameter"]["dt"]
                )
                - transient_steps
                - 1
            )
            logger.warning(
                f"Transient steps + training steps + 1 is larger than the length of training data. Reducing the number of training steps to {training_steps} (the largest possible)."
            )

        # Load the training data
        if self["Data"]["Usage"]["fileformat"] == ".npz":
            training_data = np.load(
                self.get_data_dir() / Path(f"TrainingData{index}.npz")
            )["vars"][: (transient_steps + training_steps + 1)]
        elif self["Data"]["Usage"]["fileformat"] == ".npy":
            training_data = np.load(
                self.get_data_dir() / Path(f"TrainingData{index}.npy")
            )[: (transient_steps + training_steps + 1), np.newaxis, :]
        else:
            raise ValueError(
                f"config['Data']['Usage']['fileformat'] needs to be either '.npz' or '.npy' but is {self['Data']['Usage']['fileformat']}"
            )
        logger.log(loglevel_info_single_run, "Done.")
        return training_data

    def load_evalaluation_datasets(self) -> list[np.ndarray]:
        """Load and return the evaluation data.

        Args:
            config: The configuration object.

        Returns:
            A list of numpy arrays containing the first evaluation data sets.

        Warning:
            This function should be modified to load only the amount of variables needed, to save memory.
        """
        logger.log(loglevel_info_single_run, "Loading evaluation data.")
        transient_steps = int(
            self["Data"]["Usage"]["tansient_length"]
            // self["Data"]["Creation"]["TemporalParameter"]["dt"]
        )
        evaluation_steps = int(
            self["Data"]["Usage"]["evaluation_length"]
            // self["Data"]["Creation"]["TemporalParameter"]["dt"]
        )

        # Check if the number of evaluation steps is larger than the length of the evaluation data
        if self["Data"]["model_name"] == "AlievPanfilov":
            if transient_steps + evaluation_steps + 1 > (
                self["Data"]["Creation"]["TemporalParameter"]["evaluation_length"]
                // self["Data"]["Creation"]["TemporalParameter"]["dt"]
            ):
                training_steps = (
                    (
                        self["Data"]["Creation"]["TemporalParameter"][
                            "evaluation_length"
                        ]
                        // self["Data"]["Creation"]["TemporalParameter"]["dt"]
                    )
                    - transient_steps
                    - 1
                )
                logger.warning(
                    f"Transient steps + training steps + 1 is larger than the length of training data. Reducing the number of training steps to {training_steps} (the largest possible)."
                )
        elif self["Data"]["model_name"] == "KuramotoSivashinsky":
            if transient_steps + evaluation_steps + 1 > (
                self["Data"]["Creation"]["TemporalParameter"]["evaluation_length"]
                // self["Data"]["Creation"]["TemporalParameter"]["dt"]
            ):
                training_steps = (
                    (
                        self["Data"]["Creation"]["TemporalParameter"][
                            "evaluation_length"
                        ]
                        // self["Data"]["Creation"]["TemporalParameter"]["dt"]
                    )
                    - transient_steps
                    - 1
                )
                logger.warning(
                    f"Transient steps + training steps + 1 is larger than the length of training data. Reducing the number of training steps to {training_steps} (the largest possible)."
                )

        # Load the evaluation data
        if self["Data"]["Usage"]["fileformat"] == ".npz":
            eval_data = [
                np.load(self.get_data_dir() / Path(f"EvaluationData{i}.npz"))["vars"][
                    : transient_steps + evaluation_steps + 1
                ]
                for i in range(self["Data"]["Usage"]["evaluation_datasets"])
            ]
        elif self["Data"]["Usage"]["fileformat"] == ".npy":
            eval_data = [
                np.load(self.get_data_dir() / Path(f"EvaluationData{i}.npy"))[
                    : transient_steps + evaluation_steps + 1, np.newaxis, :
                ]
                for i in range(self["Data"]["Usage"]["evaluation_datasets"])
            ]
        else:
            raise ValueError(
                f"config['Data']['Usage']['fileformat'] needs to be either '.npz' or '.npy' but is {self['Data']['Usage']['fileformat']}"
            )
        logger.log(loglevel_info_single_run, "Done.")
        return eval_data

    def parse_YAML(self) -> dict:
        """Parse the config.yml file to get systems parameters

        Returns:
            :class:`dict`: representation of the config as given in the YAML file
        """
        with open(self.path, "r") as file:
            params = yaml.safe_load(file)
        return params

    def __str__(self) -> str:
        """Return configuration as string, e.g. for printing"""
        conf_file = yaml.dump(self.dict, Dumper=IndentDumper, sort_keys=False)
        return conf_file

    def __getitem__(self, key: str):
        """Subscript access to config dict params"""
        return self.dict[key]

    def write_metadata_HDF(
        self,
        f: h5py.File,
        *,
        keylist: list[str] = ["Simulation"],
        task_id: int | None = None,
        sub_task_id: int | None = None,
    ) -> None:
        """Write parameters to an open HDF5-file's attributes

        Args:
            f (h5py.File):
                HDF file in which metadata is to be written
            keylist (list[str]):
                list of keys to be written (default: Simulation)
            task_id (int):
                if a task_id is given, the corresponding cluster params are also written
            sub_task_id (int):
                if a sub_task_id is given, only the corresponding cluster params are written
        """
        # write constant parameters
        for k in keylist:
            for key, value in self[k].items():
                f.attrs[key] = value

        # if cluster index is given, write cluster parameters as well
        if task_id is not None:
            if sub_task_id is not None:
                for key, value in self.param_scan_list()[task_id][sub_task_id].items():
                    f.attrs[key] = value
            else:
                for id in range(len(self.param_scan_list()[task_id])):
                    for key, value in self.param_scan_list()[task_id][id].items():
                        f.attrs[key] = value

    def param_scan_list(self) -> list[list[dict]]:
        r"""Return set of parameters for a cluster run

        Returns:
            list of lists of dictionaries or dictionary, where each sublist is to be
            understood as a single job in a job array

        Note:
            In order to run a simulation as a parameter scan, supply the Config YAML
            with the following block:

            .. code:: YAML

                ParamScan:
                    A:
                        - "range"
                        - [3]
                    B:
                        - "range"
                        - [5, 10]
                    C:
                        - "range"
                        - [0, 100, 10]
                    D:
                        - "list"
                        - [1, 5, 13]

                # useful shorthand for the above config:
                ParamScan:
                    A: ["range", [3]]
                    B: ["range", [5, 10]]
                    C: ["range", [0, 100, 10]]
                    D: ["list", [1, 5, 13]]

            The arguments to the parameters must be lists where the first entry
            specifies the type of parameters to expect and the second specifies the
            values.
            If the specified as :code:`"list"` then the values will simply be taken as
            specified.
            If instead specified as :code:`"range"` the values are passed to
            :func:`numpy.arange` by list unpacking, e.g. :code:`np.arange(*B)`.
            Similar commands with :code:`linspace, geomspace` are available, then the
            specified values would have to be :code:`start, stop, number`.

            The above example would mean that

            .. math::
                A \in \{0, 1, 2\} \,,
                B \in \{5, 6, 7, 8, 9\} \,,
                C \in \{0, 10, 20, \ldots, 90\} \,,
                D \in \{1, 5, 13\} \,.

            A job will then be started for each permutation of :math:`A,B,C,D`.

            If you intend to run a single execution of a fixed set of parameters, you
            may set the following within the YAML

            .. code:: YAML

                ParamScan: null
        """
        param_dict = {}

        # parse constant variables
        for parameter, value in self["Parameters"].items():
            param_dict[parameter] = [value]

        # if no ParamScan parameters are defined simply run a single execution
        if self["ParamScan"] is None:
            return [[self["Parameters"]]]

        # parse parameter scan variables
        for parameter, specs in self["ParamScan"].items():
            param_dict.pop(parameter, None)
            if specs[0] == "list":
                param_dict[parameter] = specs[1]
            elif specs[0] == "range":
                param_dict[parameter] = np.arange(*specs[1])
            elif specs[0] == "linspace":
                param_dict[parameter] = np.linspace(*specs[1])
            elif specs[0] == "geomspace":
                param_dict[parameter] = np.geomspace(*specs[1])
            else:
                raise ValueError(f"Must specify the type of parameter {parameter}")

        # generate all permutations of parameter scan values and write a list of
        # parameters wherein each entry is a single job of the simulation
        workload_permutations = itertools.product(*param_dict.values())
        workload_length = len(list(workload_permutations))
        # ensure that no sublist is empty
        if workload_length < self.max_length:
            self.max_length = workload_length
        # get a list of sublists, each being one job on the cluster
        params = [
            [dict(zip(param_dict.keys(), v)) for v in sublist]
            for sublist in divide(
                self.max_length, itertools.product(*param_dict.values())
            )
        ]
        return params

    def param_scan_len(self) -> int:
        """Total iterations of a cluster run

        Warning:
            This is **deprecated**! Use :code:`len(conf.param_scan_list())` instead
        """
        warnings.warn("use len(Config.param_scan_list()) instead", DeprecationWarning)
        return len(self.param_scan_list())

    def jobscript_datadir(self, output_type: str) -> Path:
        """Path to raw data as defined in YAML

        The cluster will expect data to be within a YAML format like such:

        args:
            output_type: The output type (:codes: `'ValidTimes', 'RunTimes', 'Memory'`), which defines the folder to be written into.

        .. code:: yaml

            Saving:
                OutputDirectory: "path/to/data"
        """
        if not (output_type in ["ValidTimes", "RunTimes", "Memory"]):
            raise ValueError(
                f"outputs must be one of 'ValidTimes', 'RunTimes', 'Memory', but is {output_type}"
            )
        outpath = (
            Path(f"Data/{output_type}")
            / Path(f"{self['Data']['model_dimension']}D_{self['Data']['model_name']}")
            / self.name
        )
        return self.get_git_root() / Path(outpath)

    def make_jobscript_datadir(
        self, *, output_type: str, copy_yaml: bool = False
    ) -> None:
        """Create datadir as specified in Config if it doesn't exist

        Args:
            output_type (str): The output type (:codes: `'ValidTimes', 'RunTimes', 'Memory'`), which defines the folder to be written into.
            copy_yaml: If set to True, the YAML will be copied to the directory created

        Warning:
            If the directory already exists, an error will be raised to avoid
            overwriting previous data.
            The recommended procedure is to either delete old data or rename the YAML
            such that new data is written in a new directory.
        """
        if not (output_type in ["ValidTimes", "RunTimes", "Memory"]):
            raise ValueError(
                f"outputs must be one of 'ValidTimes', 'RunTimes', 'Memory', but is {output_type}"
            )

        logger.log(
            loglevel_info_multiple_run,
            f"Generating ouput directory ({self.jobscript_datadir(output_type)})...",
        )
        self.jobscript_datadir(output_type).mkdir(parents=True, exist_ok=False)
        if copy_yaml:
            new_path = self.jobscript_datadir(output_type) / self.path.name
            copy(self.path, new_path)

    def generate_submission_script_from_YAML(
        self, *, output_type: str, template: Path | None = None
    ) -> Path:
        r"""Creates a shell script that can then be passed to qsub based on the
        information contained in the Config.

        Args:
            output_type (str): The output type (:code:`'ValidTimes', 'RunTimes', 'Memory'`), which defines the folder to be written into.
            template (:class:`pathlib.Path` or None):
                Optional path to a different template than
                :code:`drrc/templates/qsub.template`

        Note:
            For this to work, the configuration YAML must contain the following block:

            .. code:: YAML

                Jobscript:
                  # for qsub
                  Type: "qsub"                 # specify to run at MPI-DS
                  Cores: 4                     # number of cores per job (should be divisor of 32)
                  max_job_count: 1000          # array job will have 1000 jobs
                  # optional:
                  force_parallel_queue: False  # force job to run on teutates-mvapich2.q (optional)

                  # for slurm
                  Type: "sbatch"         # specify submission command (template must have matching name!)
                  max_job_count: 2000    # 2000 jobs will be submitted as an array
                  tasks_per_job: 4       # each job will have 4 job steps
                  cores_per_task: 4      # each job step will use 4 cores
                  mem_per_task: 24       # and 24GB of memory
                  cluster: "GWDG"        # use specific cluster options
                  time: "00:05:00"       # each task will run at most 5 minutes

            This will create a parallel jobs with 4 cores per task.
            On qsub this will always fill nodes of 32 cores.
            When using SLURM this additionally allows to set how many tasks should be
            run in each job (potentially allowing for smaller jobs / faster queueing).

            The resulting shell script will always be placed in the data directory as
            returned by :func:`Config.jobscript_datadir` to ensure data is kept with the
            submission script.

        Warning:
            There seems to be some configuration in place for :code:`cluster: "raven"`
            and :code:`cluster: "viper"` such that total memory must always be defined.
            I'm not quite sure yet how to write scripts for that.
            **So right now this only works @GWDG!**

        Important:
            **If using** :code:`Type: "qsub"`:

            Currently this is set up such that a single cluster node of 32 cores will
            receive as many jobs as it can fit. For optimal use of the cluster
            :code:`Cores` should be a divisor of 32.

            When choosing this, keep in mind that per cluster node there are 192GB RAM.

            If :code:`Cores` is set to 1, this function assumes that the job will be
            submitted to the serial cluster and thus adapt the submission script.
            However, one may set the optional argument
            :code:`force_parallel_queue: True` to run 32 single-core jobs per node in
            the parallel queue.

            Jobs are submitted using:

            .. code:: bash

                qsub Submit-job.sh


            **If using** :code:`Type: "slurm"`:

            In this case :code:`RAM, partition` must be defined in the YAML.
            Note that this is the RAM per CPU core. So the in the above example SLURM
            will allocate :math:`4\cdot12\mathrm{GB}=48\mathrm{GB}` of RAM.

            The job can then be submitted using

            .. code:: bash

                sbatch Submit-job.sh
        """
        from jinja2 import Template

        logger.log(loglevel_info_single_run, "Generating submission script...")

        script_name = f"{self.jobscript_datadir(output_type)}/Submit-{self.name}.sh"

        # allow for different workload managers
        template_name = self["Jobscript"]["Type"]

        # read template
        if template is None:
            template_path = (
                Path(__file__).parent / "templates" / f"{template_name}.template"
            )
        else:
            template_path = Path(template)
        logger.log(loglevel_info_multiple_run, f"found template at {template_path}")
        with open(template_path, "r") as f:
            script_template = f.read()

        # template variables
        if template_name == "qsub":
            # get optional parameters from YAML
            fpq_value = self["Jobscript"].get("force_parallel_queue")
            fpq = fpq_value if fpq_value is not None else False

            # collect all placeholder values
            script_vars = {
                "JOB_NAME": self.name,
                "OUTPUT_PATH": self.jobscript_datadir(output_type),
                "GIT_ROOT": self.get_git_root(),
                "EXECUTABLE": Path(sys.argv[0]).absolute(),
                "NUM_CORES": self["Jobscript"]["Cores"],
                "YAMLPATH": self.path,
                "JOB_LENGTH": self.param_scan_len(),
                "JOB_STRIDE": CLUSTER_CORES // self["Jobscript"]["Cores"],
                # optional arguments
                "FORCE_PARALLEL_QUEUE": fpq,
            }
        elif template_name == "sbatch":
            # calculate total memory
            mem_total = int(
                self["Jobscript"]["tasks_per_job"] * self["Jobscript"]["mem_per_task"]
            )
            mem_per_cpu = int(
                self["Jobscript"]["mem_per_task"] / self["Jobscript"]["cores_per_task"]
            )
            script_vars = {
                "JOB_NAME": self.name,
                "NTASKS": self["Jobscript"]["tasks_per_job"],
                "OUTPUT_PATH": self.jobscript_datadir(output_type),
                "GIT_ROOT": self.get_git_root(),
                "EXECUTABLE": Path(sys.argv[0]).absolute(),
                "NCORES": self["Jobscript"]["cores_per_task"],
                "YAMLPATH": self.path,
                "JOB_LENGTH": self["Jobscript"]["max_job_count"],
                "MEM": mem_total,
                "MEM_PER_CPU": mem_per_cpu,
                "CLUSTER": self["Jobscript"]["cluster"],
                "TIME": self["Jobscript"]["time"],
            }
        else:
            raise ValueError('must define a valid template name as "Type"!')

        # populate template
        populated = Template(script_template).render(script_vars)

        # write to file & make script executable
        with open(script_name, "w") as f:
            f.write(populated)
        Path(script_name).chmod(0o755)
        logger.log(loglevel_info_multiple_run, f"Generated {script_name}")

        return Path(script_name).absolute()
