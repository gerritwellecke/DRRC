""" This script is created for the cluster run of the parallel reservoirs. 
It is used to return the proper output directory and evaluate multiple parallel reservoirs on the cluster.

.. codeauthor:: Luk Fleddermann and Gerrit Wellecke
"""

import argparse
import logging as log
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from drrc.config import Config
from drrc.parallelreservoirs import (
    ParallelReservoirs,
    ParallelReservoirsArguments,
    ParallelReservoirsFFT,
    ParallelReservoirsPCA,
)
from drrc.tools.logger_config import drrc_logger as logger

# set up logging
loglevel_info_multiple_run = log.getLevelName("INFO_nRUN")
logger.propagate = False
logger.setLevel(log.getLevelName("INFO_nRUN"))  # INFO_nRUN and INFO_1RUN


def concatenate_csv_files(glob_pattern) -> pd.DataFrame:
    """
    Concatenates multiple CSV files into a single DataFrame.

    Args:
        glob_pattern (str): The glob pattern used to match CSV files.

    Returns:
        pandas.DataFrame: A DataFrame resulting from the concatenation of all matched
            CSV files, or None if no files match the glob pattern.

    Raises:
        ValueError: If the glob pattern does not match any files.
    """
    # Get a list of all the csv files matching the glob pattern
    csv_files = glob(glob_pattern)

    # Concatenate all the dataframes into one
    concatenated_df = pd.concat(
        (pd.read_csv(fp, index_col=0) for fp in csv_files), ignore_index=True
    )

    return concatenated_df


def main():
    # argparser, get the index of simulation & yaml file path
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml", type=str, default=None)
    parser.add_argument("job_idx", type=int, default=None)
    args = parser.parse_args()
    job_idx = args.job_idx - 1

    # load configuration
    conf = Config(Path(args.yaml).absolute())
    # logger.log(loglevel_info_multiple_run, f"Job index: {job_idx}")
    if int(job_idx) == -1:  # return jobscriptpath to initialize the array job
        # for quick debugging
        logger.log(
            loglevel_info_multiple_run,
            f"Array job with {conf.param_scan_len()} executions.\n\nCONFIGURATION:\n",
        )
        logger.log(loglevel_info_multiple_run, str(conf))

        # ensure that datadir exists
        conf.make_jobscript_datadir(output_type="ValidTimes", copy_yaml=True)
        # generate the cluster submission script
        jobscript_path = conf.generate_submission_script_from_YAML(
            output_type="ValidTimes"
        )
        print(conf["Jobscript"]["Type"], jobscript_path)

    else:  # run the actual job
        # load constant parameters from YAML
        parameter_list = conf.param_scan_list()[job_idx]
        ### START CHERRY PICKING #######################################################
        parameter_df = pd.DataFrame.from_dict(parameter_list)

        df = concatenate_csv_files("Data/Reservoir/**/DataFrame.csv").dropna()
        # df = concatenate_csv_files("../../Data/Reservoir/**/DataFrame.csv").dropna()
        # find maximum valid time for each DR_type, DR_ratio, num_res
        dfgrouped = df.loc[
            df.groupby(["nodes", "DR_type", "DR_ratio", "num_res"])["mean_t"].idxmax()
        ]
        # use new key names
        dfgrouped = dfgrouped.rename(
            columns={
                "nodes": "reservoir_nodes",
                "spectral": "adjacency_spectralradius",
                "inscale": "input_scaling",
                "leakage": "reservoir_leakage",
                "regularization": "training_regularization",
                "DR_type": "Transformation",
                "DR_ratio": "dimensionreduction_fraction",
                "num_res": "parallelreservoirs_grid_shape",
            }
        )

        # exclude FFT
        dfgrouped = dfgrouped[
            (dfgrouped["Transformation"] == "identity")
            | (dfgrouped["Transformation"] == "pca")
        ]

        # remove cherry picked parameters from YAML parameters
        parameter_df.drop(
            columns=dfgrouped.keys().intersection(parameter_df.keys()), inplace=True
        )

        # make new list of dictionaries
        parameter_list = (
            dfgrouped.merge(parameter_df, how="cross")
            .astype({"parallelreservoirs_grid_shape": int})
            .to_dict("records")
        )
        ### END CHERRY PICKING #########################################################

        # load proper training and eval data
        evaluation_data = conf.load_evalaluation_datasets()
        training_data = conf.load_training_dataset(
            index=parameter_list[0]["training_data_index"]
        )
        trainings_index = parameter_list[0]["training_data_index"]

        # iterate over the parameter list
        logger.log(
            loglevel_info_multiple_run,
            f"Starting job {job_idx+1} with {len(parameter_list)} subtasks",
        )

        # this list will collect DataFrames of each single_run
        output_list = []

        for i, parameters in enumerate(parameter_list):
            # update training data if necessary
            if parameter_list[i]["training_data_index"] != trainings_index:
                training_data = conf.load_training_dataset(
                    index=parameter_list[0]["training_data_index"]
                )
                trainings_index = parameter_list[i]["training_data_index"]

            # run the single task
            validsteps, seed = single_run(
                conf=conf,
                job_idx=job_idx,
                sub_idx=i,
                parameter_dict=parameters,
                training_data=training_data,
                evaluation_data=evaluation_data,
                transient_steps=int(
                    conf["Data"]["Usage"]["tansient_length"]
                    // conf["Data"]["Creation"]["TemporalParameter"]["dt"]
                ),
            )
            # write seed as string to ensure no rounding will take place
            parameter_list[i]["seed"] = str(seed)

            # generate DataFrame with parameters
            params_df = pd.DataFrame.from_dict(parameter_list[i])

            # generate Series from valid time results
            t_val_series = pd.Series(
                validsteps * conf["Data"]["Creation"]["TemporalParameter"]["dt"],
                name="valid_time",
            )

            # cross merge the parameters with the valid time results
            df_out = params_df.merge(t_val_series, how="cross")
            output_list.append(df_out)

        # gather all DataFrames
        df = pd.concat(output_list)
        # write output DF to file
        df.to_csv(
            conf.jobscript_datadir(output_type="ValidTimes")
            / f"validtimes-{job_idx+1}.csv",
            index=False,
        )

        logger.log(loglevel_info_multiple_run, f"Finished job {job_idx+1}")


def single_run(
    conf,
    job_idx,
    sub_idx,
    parameter_dict,
    training_data,
    evaluation_data,
    transient_steps,
) -> tuple[np.ndarray, int]:
    """
    Run a single task.
    That means: Evaluate a single set on parameters from the list of dictionaries returned by the Config object on a fixed number (specified in the conf object) of evaluation data files.

    """

    # get length of training/evaluation/transient data
    step_size = conf["Data"]["Creation"]["TemporalParameter"]["dt"]
    transient_steps = int(conf["Data"]["Usage"]["tansient_length"] // step_size)
    evaluation_steps = int(conf["Data"]["Usage"]["evaluation_length"] // step_size)

    # set up reservoir parameter
    par = ParallelReservoirsArguments.from_dict(parameter_dict)

    # initialize the reservoir
    transformation_training_path = (
        f"{conf['Data']['model_dimension']}D_{conf['Data']['model_name']}"
    )
    if conf["Transformation"] == "pca":
        res = ParallelReservoirsPCA(
            Parameter=par, prediction_model=transformation_training_path
        )
    elif conf["Transformation"] == "fft":
        res = ParallelReservoirsFFT(
            Parameter=par, prediction_model=transformation_training_path
        )
    elif conf["Transformation"] == "identity":
        res = ParallelReservoirs(Parameter=par)
    else:
        raise ValueError(
            f"dictionary['Transformation'] needs to be either 'pca', 'fft' or 'identity' but is {conf['Transformation']}"
        )

    # train the reservoir
    res.train(
        input_training_data=training_data[:-1],
        output_training_data=training_data[1 + transient_steps :],
        transient_steps=transient_steps,
    )
    valid_steps = np.full(conf["Data"]["Usage"]["evaluation_datasets"], np.nan)

    logger.log(
        loglevel_info_multiple_run,
        f"Starting evaluation of job {job_idx+1} subtask {sub_idx+1}",
    )
    for evaluation_nr in range(conf["Data"]["Usage"]["evaluation_datasets"]):
        # transient update
        res.reservoir_transient(
            input=evaluation_data[evaluation_nr][:transient_steps],
            predict_on_transient=False,
        )
        # iteratively predict a timeseries
        _, _, valid_steps[evaluation_nr] = res.iterative_predict(
            initial=evaluation_data[evaluation_nr][
                transient_steps : transient_steps + 1
            ],  # this is the first step only. Just keeping temporal dimension
            max_steps=evaluation_steps,
            supervision_data=evaluation_data[evaluation_nr][
                transient_steps + 1 : transient_steps + evaluation_steps + 1
            ],
            error_function="NRMSE",
            mean_norm=conf["Data"]["Usage"]["mean_norm"],
            error_stop=conf["Data"]["Usage"]["error_stop"],
        )
    logger.log(
        loglevel_info_multiple_run,
        f"Finished evaluation of job {job_idx+1} subtask {sub_idx+1}",
    )
    # return the results
    return valid_steps, res.seed


if __name__ == "__main__":
    main()
