""" This script is created for the cluster run of the parallel reservoirs. 
It is used to return the proper output directory and evaluate multiple parallel reservoirs on the cluster.

.. codeauthor:: Luk Fleddermann and Gerrit Wellecke
"""
import argparse
import logging as log
import time as t
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


def main(comp_time_dict, start_time):
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
        # Restrict RunTime profiling to jobs with one subtask and one evaluation dataset
        if len(conf.param_scan_list()[0]) > 1:
            raise ValueError(
                "Runtime profiling jobs are not intended for multiple subtasks per job. Modify total number of tasks or maximum number of array jobs."
            )
        if not conf["Data"]["Usage"]["evaluation_datasets"] == 1:
            raise ValueError(
                "Runtime profiling jobs are not intended for multiple evaluation datasets. Modify total number of evaluation datasets."
            )

        # for quick debugging
        logger.log(
            loglevel_info_multiple_run,
            f"Array job with {conf.param_scan_len()} executions.\n\nCONFIGURATION:\n",
        )
        logger.log(loglevel_info_multiple_run, str(conf))

        # ensure that datadir exists
        conf.make_jobscript_datadir(output_type="RunTimes", copy_yaml=True)
        # generate the cluster submission script
        jobscript_path = conf.generate_submission_script_from_YAML(
            output_type="RunTimes"
        )
        print(conf["Jobscript"]["Type"], jobscript_path)

    else:  # run the actual job
        parameter_list = conf.param_scan_list()[job_idx]

        # load proper training and eval data
        tmp = t.time()
        evaluation_data = conf.load_evalaluation_datasets()
        training_data = conf.load_training_dataset(
            index=parameter_list[0]["training_data_index"]
        )
        trainings_index = parameter_list[0]["training_data_index"]
        comp_time_dict["DataLoadTime"] = t.time() - tmp

        # iterate over the parameter list
        logger.log(
            loglevel_info_multiple_run,
            f"Starting job {job_idx+1} with {len(parameter_list)} subtasks",
        )

        # this list will collect DataFrames of each single_run
        output_list = []

        for i in range(len(parameter_list)):
            # update training data if necessary
            if parameter_list[i]["training_data_index"] != trainings_index:
                training_data = conf.load_training_dataset(
                    index=parameter_list[0]["training_data_index"]
                )
                trainings_index = parameter_list[i]["training_data_index"]

            logger.log(
                loglevel_info_multiple_run,
                f" Using dictionary: {str(parameter_list[i])}",
            )

            # run the single task
            runtimes, seed = single_run(
                conf=conf,
                job_idx=job_idx,
                sub_idx=i,
                training_data=training_data,
                evaluation_data=evaluation_data,
                transient_steps=int(
                    conf["Data"]["Usage"]["tansient_length"]
                    // conf["Data"]["Creation"]["TemporalParameter"]["dt"]
                ),
                comp_time_dict=comp_time_dict,
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
            df_params = pd.DataFrame.from_dict(parameter_list[i]).T

            # create runtime dataframe
            comp_time_dict["total_RunTime"] = t.time() - start_time
            df_times = pd.DataFrame(runtimes, index=[0]).T

            # append to output list
            output_list.append(df_params)
            output_list.append(df_times)

        # gather all DataFrames
        # df = pd.concat(output_list)
        df_combined = pd.concat(output_list, axis=0).T

        # write output DF to file
        df_combined.to_csv(
            conf.jobscript_datadir(output_type="RunTimes")
            / f"runtimes-{job_idx+1}.csv",
            index=False,
        )

        logger.log(loglevel_info_multiple_run, f"Finished job {job_idx+1}")


def single_run(
    conf,
    job_idx,
    sub_idx,
    training_data,
    evaluation_data,
    transient_steps,
    comp_time_dict,
) -> tuple[np.ndarray, int]:
    """
    Run a single task.
    That means: Evaluate a single set on parameters from the list of dictionaries returned by the Config object on a fixed number (specified in the conf object) of evaluation data files.

    """
    tmp = t.time()
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
    comp_time_dict["ReservoirInitTime"] = t.time() - tmp

    # train the reservoir
    tmp = t.time()
    res.train(
        input_training_data=training_data[:-1],
        output_training_data=training_data[1 + transient_steps :],
        transient_steps=transient_steps,
    )
    comp_time_dict["ReservoirTrainTime"] = t.time() - tmp

    valid_steps = np.full(conf["Data"]["Usage"]["evaluation_datasets"], np.nan)

    logger.log(
        loglevel_info_multiple_run,
        f"Starting evaluation of job {job_idx+1} subtask {sub_idx+1}",
    )
    for evaluation_nr in range(conf["Data"]["Usage"]["evaluation_datasets"]):
        # transient update
        tmp = t.time()
        res.reservoir_transient(
            input=evaluation_data[evaluation_nr][:transient_steps],
            predict_on_transient=False,
        )

        # Transient update time per step
        comp_time_dict["TransientUpdateTime"] = (t.time() - tmp) / transient_steps

        # iteratively predict a timeseries
        tmp = t.time()
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
            error_stop=0,
            extra_steps=100,
        )
        # Prediction time per step (always 100 steps are excecuted)
        comp_time_dict["PredictionTime"] = (t.time() - tmp) / 100
    logger.log(
        loglevel_info_multiple_run,
        f"Finished evaluation of job {job_idx+1} subtask {sub_idx+1}",
    )
    # return the results
    return comp_time_dict, res.seed


if __name__ == "__main__":
    start = t.time()
    computation_time_dict = {}
    main(computation_time_dict, start)
