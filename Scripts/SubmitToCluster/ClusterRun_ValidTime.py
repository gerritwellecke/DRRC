""" This script is created for the cluster run of the parallel reservoirs. 
It is used to return the proper output directory and evaluate multiple parallel reservoirs on the cluster.

.. codeauthor:: Luk Fleddermann and Gerrit Wellecke
"""

import argparse
import logging as log
from pathlib import Path
from time import time

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


def write_output(output_list: list, job_idx, conf: Config) -> None:
    # gather all DataFrames
    df = pd.concat(output_list)
    # write output DF to file
    df.to_csv(
        conf.jobscript_datadir(output_type="ValidTimes")
        / f"validtimes-{job_idx+1}.csv",
        index=False,
    )


def single_run(
    conf, job_idx, sub_idx, training_data, evaluation_data, transient_steps
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
    par = ParallelReservoirsArguments.from_config(
        conf=conf, job_idx=job_idx, sub_idx=sub_idx
    )

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
            f"Array job with {conf.param_scan_len()} executions ({len(conf.param_scan_list()[0])} subtasks each).\n\nCONFIGURATION:\n",
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
        parameter_list = conf.param_scan_list()[job_idx]

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

        time_last_save = time()
        for i in range(len(parameter_list)):
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
                training_data=training_data,
                evaluation_data=evaluation_data,
                transient_steps=int(
                    conf["Data"]["Usage"]["tansient_length"]
                    // conf["Data"]["Creation"]["TemporalParameter"]["dt"]
                ),
            )
            # write seed as string to ensure no rounding will take place
            parameter_list[i]["seed"] = str(seed)

            # cast tuples to list of tuples for proper saving
            parameter_list[i]["identical_outputmatrix"] = [
                parameter_list[i]["identical_outputmatrix"]
            ]
            parameter_list[i]["spatial_shape"] = [parameter_list[i]["spatial_shape"]]
            parameter_list[i]["parallelreservoirs_grid_shape"] = [
                parameter_list[i]["parallelreservoirs_grid_shape"]
            ]

            # write transformation to parameter list
            parameter_list[i]["Transformation"] = conf["Transformation"]

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

            # ensure that output is written at least every 30 minutes
            if (time() - time_last_save) > 1800:
                time_last_save = time()
                write_output(output_list, job_idx, conf=conf)
                logger.log(
                    loglevel_info_multiple_run, f"Saving data after subtask {i+1}"
                )

        # write final output
        write_output(output_list, job_idx, conf=conf)

        logger.log(loglevel_info_multiple_run, f"Finished job {job_idx+1}")


if __name__ == "__main__":
    main()
