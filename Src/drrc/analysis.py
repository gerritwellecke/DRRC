import logging
import os
from abc import ABC, abstractmethod
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from drrc.config import Config


class AutomaticPostprocessing:
    """This class enables the automatic concatenation of all subtask outputs into a
    single file called "DataFrame.csv".
    Within the respective job directories this still generates a unique path.
    """

    def __init__(self, root: Path):
        """Initialise automatic concatenation

        Args:
            root:
                most shallow path from which to look for job directories, starting from
                git root

        Todo:
            - make this work with relative paths from the git root
            - either use a root path or supply a yaml for single job post processing
        """
        self.path = root

        cpus = os.cpu_count()
        if cpus is not None:
            self.num_cores = cpus // 2
        else:
            self.num_cores = 1

    def summary(self) -> None:
        """Print summary of the automatic concatenation

        Note:
            This function does not touch any of the files!
            Its main purpose is debugging / checking that
            :func:`AutomaticConcatenation.auto_concatenate` behaves as expected.
        """
        relevant_paths = [p.parent for p in Path(self.path).glob("**/*-1.csv")]
        print(f"Checking in: {relevant_paths}")

        for fp in relevant_paths:
            if not (fp / Path("DataFrame.csv")).is_file():
                print(f"Concatenation needed in: {fp}")

    def auto_concatenate(self, Delete=False) -> None:
        """Automatically concatenate all raw output files

        Args:
            Delete:
                If True, then the raw output files are deleted after concatenation.
                Default is False.

        The result is then saved with the raw output in a file named
        :code:`DataFrame.csv`
        """
        logging.info(f"Concatenating data in subfolders of {self.path}")
        key = self.path.name
        # make iterator of all paths to be considered
        relevant_paths = list(Path(self.path).glob("**/*-1.csv"))
        # in each path, check if DataFrame.csv file exists
        for fp in relevant_paths:
            fp = fp.parent
            key_index = fp.parts.index(key)

            # if not, then call self._concatenate with the current path
            if not (fp / Path("DataFrame.csv")).is_file():
                logging.info(f"{Path(*fp.parts[key_index:])}/DataFrame.csv' not found")
                df = self._concatenate(fp)

                # Check if both 'transformation' and 'Transformation' exist
                if "transformation" in df.columns and "Transformation" in df.columns:
                    # If both exist, drop 'transformation'
                    df.drop("transformation", axis=1, inplace=True)
                elif "transformation" in df.columns:
                    # If only 'transformation' exists, rename it to 'Transformation'
                    df.rename(
                        columns={"transformation": "Transformation"}, inplace=True
                    )
                # If only 'Transformation' exists, do nothing

                df.to_csv(fp / Path("DataFrame.csv"), index=False)
                logging.info(
                    f"Saved concatenated data at {Path(*fp.parts[key_index:])}/DataFrame.csv"
                )
            else:
                logging.info(
                    f"{Path(*fp.parts[key_index:])}/DataFrame.csv' already exists"
                )

            if Delete:
                # delete all raw output files
                for f in fp.glob("*-*.csv"):
                    f.unlink()

    def _concatenate(self, path: Path) -> pd.DataFrame:
        """Concatenate a single cluster job

        This takes all numbered :code:`.csv` DataFrames and concatenates them into one.
        It checks if the number of lines agrees with what is expected from the config. Else it raises a warning.

        Args:
            path: path to output of cluster job

        Returns:
            :class:`pd.DataFrame` which contains all data from a single job
        """
        # get expected number of files
        conf = Config(list(path.glob("*.yml"))[0])
        tasks = [
            len(pars) * conf["Data"]["Usage"]["evaluation_datasets"]
            for pars in conf.param_scan_list()
        ]
        total_tasks = np.sum(tasks)
        # get all relevant paths
        relevant_paths = list(Path(path).glob("*-*.csv"))

        # collect data into new DataFrame
        # df = pd.concat([pd.read_csv(f) for f in tqdm(relevant_paths, desc=f"Concatenating {str(path.name)}")])

        # iterate through all score files in parallel & write statistics
        stat_list = []
        with Pool(processes=self.num_cores) as pool:
            logging.info(
                f"Processing {len(relevant_paths)} files in {path.name} on {self.num_cores} cores."
            )
            stat_list += pool.starmap(
                self._read_test_csv, [(path, tasks) for path in relevant_paths]
            )

        # return the DataFrame
        df = pd.concat(stat_list, ignore_index=True)
        if len(df) != total_tasks:
            logging.warning(
                f"Jobs died in {path.name}! Length of csv is {len(df)}, which does not match expected number of total entries {total_tasks}."
            )
        return df

    def _read_test_csv(self, path: Path, jobs: list) -> pd.DataFrame:
        """Read a single csv file, test if it has the expected number of results and return the DataFrame.

        Args:
            path: path to csv file
            jobs: list of number of jobs per task

        Returns:
            :class:`pd.DataFrame` which contains all data from a single job
        """

        df = pd.read_csv(path)
        # get zero based task id from 1 based file name
        task_id = int(str(path.name).split("-")[-1].split(".")[0]) - 1
        if len(df) != jobs[task_id]:
            logging.warning(
                f"Number of entries in {path.parent.name}/{path.name} is {len(df)} and does not match expected number of entries {jobs[task_id]}."
            )
        return df

    def auto_statisticsgeneration(self) -> None:
        """Generate statistics from the concatenated DataFrame"""

        logging.info(f"Generating statistics of data in subfolders of {self.path}")
        key = self.path.name
        # make iterator of all paths to be considered
        relevant_paths = list(Path(self.path).glob("**/DataFrame.csv"))

        # in each path, check if DataFrame.csv file exists
        for fp in relevant_paths:
            fp = fp.parent
            key_index = fp.parts.index(key)
            # if not, then call self._concatenate with the current path
            if not (fp / Path("ProcessedValidTimes.csv")).is_file():
                logging.info(
                    f"{Path(*fp.parts[key_index:])}/ProcessedValidTimes.csv not found"
                )
                self._generate_statistics(
                    pd.read_csv(fp / Path("DataFrame.csv"))
                ).to_csv(fp / Path("ProcessedValidTimes.csv"), index=False)
                logging.info(
                    f"Saved processed data at {Path(*fp.parts[key_index:])}/ProcessedValidTimes.csv"
                )
            else:
                logging.info(
                    f"{Path(*fp.parts[key_index:])}/ProcessedValidTimes.csv already exists"
                )

    def _generate_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate statistics from the concatenated DataFrame"""

        if "transformation" in df.keys():
            df.rename(columns={"transformation": "Transformation"}, inplace=True)

        params = [
            "adjacency_degree",
            "adjacency_dense",
            "input_bias",
            "spatial_shape",
            "system_variables",
            "parallelreservoirs_ghosts",
            "boundary_condition",
            "identical_inputmatrix",
            "identical_adjacencymatrix",
            "identical_outputmatrix",
            "training_includeinput",
            "training_output_bias",
            "adjacency_spectralradius",
            "input_scaling",
            "training_regularization",
            "reservoir_leakage",
            "parallelreservoirs_grid_shape",
            "dimensionreduction_fraction",
            "reservoir_nodes",
            "Transformation",
        ]
        return (
            df.groupby(params)["valid_time"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "mean_ValidTime", "std": "std_ValidTime"})
        )


class AnalyseClusterRunBase(ABC):
    """Base class for all analysis types we will run later.

    This includes all basic functionality that will be used later, such as IO.
    """

    def __init__(self, conf: Config):
        """Initialize a cluster run object for analysis

        Args:
            conf (Config):
                A config object that has previously been run on the cluster.
        """
        self.conf = conf

    @abstractmethod
    def process(self) -> pd.DataFrame:
        """Read output of cluster run as defined by self.conf"""
        pass

    @abstractmethod
    def save(self) -> None:
        """Save processed data in a file"""
        pass


class HyperPostProcessing(AnalyseClusterRunBase):
    """Post-processing of our hyperparameter scans

    This class is meant to take in raw data and generate the desired DataFrame
    """

    def __init__(
        self,
        conf: Config,
        data_name: str = "score_",
        data_type: str = ".txt",
        num_cores: int = os.cpu_count() // 2,
    ):
        """Initialize post-processing

        Important:
            This class assumes that output files contain a numerical (1-based) index
            between :code:`data_name` and :code:`data_type`!

        Args:
            conf (Config):
                Config of the corresponding clusterrun
            data_dir (strPath):
                Path to raw data
            out_dir (Path):
                Path for saving dataframe
            data_name (str):
                File name, e.g. :code:`"score_"`
            data_type (str):
                File extension, e.g. :code:`".txt"`
            num_cores (int):
                Number of cores available for processing
        """
        super().__init__(conf)
        self.df = None
        self.data_name = data_name
        self.data_type = data_type
        self.number_seeds = self.conf["Training"]["Datasets"]
        self.number_evals = self.conf["Evaluation"]["Datasets"]
        self.num_cores = num_cores
        self.data_dir = Path(self.conf["Saving"]["OutputDirectory"])
        self.out_dir = self.conf.get_git_root() / Path(
            self.conf["Saving"]["OutputDirectory"]
        )

    # TODO: make data_dir a property
    @property
    def data_dir(self):
        return self._data_dir

    @data_dir.setter
    def data_dir(self, data_dir: Path):
        """Set path to raw data and search three possible git repos on file server, if
        given path does not exist"""
        # Testing and Creating Directories
        self._data_dir = self.conf.get_git_root() / data_dir
        testfile = self._get_raw_file(0)

        # Find raw data in file system
        FilesNotFound = False
        try:
            _ = np.loadtxt(testfile, skiprows=13)
        except FileNotFoundError:
            logging.info(
                f"{testfile} not found in local git repository. Testing other users."
            )
            FilesNotFound = True
            GitRepos = [
                Path("/data.bmp/gwellecke/DRRC/"),
                Path("/data.bmp/khollborn/DRRC/"),
                Path(
                    "/data.bmp/lfleddermann/DataAnalysis/2023_Paper_Reservoir_DimensionReduction/DRRC/"
                ),
            ]
            i = 0
            while FilesNotFound and i < len(GitRepos):
                try:
                    _ = np.loadtxt(GitRepos[i] / testfile, skiprows=13)
                    self._data_dir = GitRepos[i] / self._data_dir
                    logging.info(f"Using {self._data_dir} as Datadirectory.")
                    FilesNotFound = False
                except FileNotFoundError:
                    logging.info(f"Files not found at {GitRepos[i]/self._data_dir}.")
                    pass
                i += 1
        if FilesNotFound:
            raise FileNotFoundError("Data does not exist.")

    @property
    def out_dir(self):
        return self._out_dir

    @out_dir.setter
    def out_dir(self, out_dir: Path):
        """Set variable and make directory for output"""
        self._out_dir = out_dir
        try:
            os.makedirs(self._out_dir)
            logging.info(f"Created Outputfolder: {self._out_dir}")
        except FileExistsError:
            logging.warning(
                f"Outputdirectory {self._out_dir} already exists. Data might be overwritten."
            )

    def _get_raw_file(self, i: int) -> Path:
        """Generate filename for the n-th raw data file

        Args:
            i (int): Number of the file using zero-based indexing

        Returns:
            (str): Path to requested file

        Todo:
            Make this method return a Path instead of str
        """
        return Path(
            str(self.data_dir) + "/" + self.data_name + str(i + 1) + self.data_type
        )

    def _generate_statistics(self, i: int, params: dict):
        """Internal function to generate the statistics of :func:`self.process`

        Warning:
            This function will not return any statistics if the numpy array contains any
            NaNs. Numpy has functions for this, but right now we don't use them.
        """
        raw_file = self._get_raw_file(i)
        try:
            # new: (works with missing data)
            col_names = ["seed"] + [f"val_{i}" for i in range(self.number_evals)]
            raw_input = pd.read_csv(
                raw_file, sep="\t", names=col_names, skiprows=1
            ).to_numpy()

            tmp_Times = raw_input[:, 1:].astype(float)  # N x M
            # we don't ever use the seed...
            # tmp_seeds = raw_input[:, 0]  # N x 1

            # BETTER APPROACH:
            # calculate statistics at this point and write to dataframe later
            # want: mean, std, max, mean_std_seed, mean_std_eval
            # --> just take param_list and add the needed keys to it!
            # i.e.:
            params["mean_t"] = tmp_Times.mean()
            params["std_t"] = tmp_Times.std()
            params["max_t"] = tmp_Times.max()
            params["avg_std_seed"] = tmp_Times.std(axis=0).mean()
            params["avg_std_data"] = tmp_Times.std(axis=1).mean()

            # write important information about cluster run
            params["DR_type"] = self.conf["Reservoir"]["transform"]
            params["DR_ratio"] = self.conf["Reservoir"]["fracin"]
            params["num_res"] = self.conf["Reservoir"]["replicas"][0]

        except (FileNotFoundError, ValueError) as error:
            logging.warning(f"{error} opening {raw_file}")

        return params

    def process(
        self,
    ) -> pd.DataFrame:
        """Extract Validtime data and seeds from :code:`.txt` files stored in
        :code:`conf['Saving']['OutputDirectory']` into a :class:`pd.DataFrame`.

        To accelerate pre-processing of raw data this function runs on half the system's
        cores by default.

        Args:
            conf (Config):
                :class:`Config` with ClusterRun information.
            DataDir (Path):
                Passed to Data, might deviate from Path in conf depending on user.
            DataName (str):
                Name of files with Validtime data up to iterator number.
                Default :code:`score_`.
            DataType (str):
                File type of files with Validtime data. Default :code:`.txt`.
            NumberSeeds (int):
                Number of seeds/ different networks drawn. Default 10.
            num_cores (int):
                Number of CPU cores to use for the preprocessing.
                Default is half the system's cores.

        Return:
            (pd.DataFrame):
                Each row corresponds to a single set of hyperparameters.
                Statistics are given for the following:

                .. list-table::

                    * - :code:`mean_t`
                      - Mean valid time over all executions of the hyperparameter set
                    * - :code:`std_t`
                      - Standard variance of :code:`mean_t` over all executions of the
                        hyperparameter set
                    * - :code:`max_t`
                      - Maximum valid time over all executions of the hyperparameter set
                    * - :code:`avg_std_seed`
                      - Average standard deviation per training seed
                    * - :code:`avg_std_data`
                      - Average standard deviation per training dataset
        """
        param_list = self.conf.param_scan_list()
        stat_list = []

        # iterate through all score files in parallel & write statistics
        with Pool(processes=self.num_cores) as pool:
            logging.info(
                f"Processing {len(param_list)} files on {self.num_cores} cores."
            )
            stat_list += pool.starmap(self._generate_statistics, enumerate(param_list))

        # return the DataFrame
        self.df = pd.DataFrame(stat_list)
        return self.df

    def save(self) -> None:
        """Save dataframe to csv"""
        self.df.to_csv(self.out_dir / Path("DataFrame.csv"))
        return

    def fix_raw_data(self) -> None:
        """Fix bad output raw data and save to new file with prefix :code:`fn_mod`

        This function also modifies the expected filename.

        Args:
            fn_mod(str):
                Modified prefix of the input file. Default is :code:`"rf"`, such that,
                e.g. :code:`"score_0.txt" --> "rf-score_0.txt"`
        """
        # loop over all raw files
        file_count = len(self.conf.param_scan_list())
        with Pool(processes=self.num_cores) as pool:
            logging.info(f"Re-formatting {file_count} files on {self.num_cores} cores.")
            pool.map(self._reformat_single_file, range(file_count))

    def _reformat_single_file(self, file_index: int):
        """Takes a filename and creates the reformatted file.

        Args:
            file_index (int):
                Index of the file to be reformatted, starting at 0.
        """
        filename = self._get_raw_file(file_index)
        old_filename = filename
        filename = filename.parent / Path("old_" + str(filename.name))

        try:
            f = open(old_filename, "r")
            seeds = []
            valtimes = []
            # skip two lines
            f.readline()
            if f.readline() == "seeds:\n":
                os.rename(old_filename, filename)
                # read seeds up to ':'
                while True:
                    line = f.readline()
                    if line == ":\n":
                        break
                    seeds.append(line[:-1])

                # then read the rest of the file, i.e. valtimes
                while True:
                    line = f.readline()
                    if not line:
                        break
                    valtimes.append(line)

                # join seeds with valtimes
                reformatted = [seeds[i] + "\t" + valtimes[i] for i in range(len(seeds))]

                f.close()

                # write reformatted file
                with open(old_filename, "w") as f:
                    f.writelines(reformatted)

                if file_index % 100 == 0:
                    logging.debug(f"Reformated {file_index}.")
            else:
                logging.debug(f"{old_filename} already formatted")

        except FileNotFoundError:
            logging.info(f"{old_filename} doesnt exist.")
