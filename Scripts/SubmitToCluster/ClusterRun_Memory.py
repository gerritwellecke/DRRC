""" This script is created for the cluster run of the parallel reservoirs. 
It is used to return the proper output directory and evaluate multiple parallel reservoirs on the cluster.

.. codeauthor:: Luk Fleddermann and Gerrit Wellecke
"""
import argparse
import logging as log
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from memory_profiler import memory_usage

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


def main(mem_dict: dict):
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
        # Restrict memory profiling to jobs with one subtask and one evaluation dataset
        if len(conf.param_scan_list()[0]) > 1:
            raise ValueError(
                "Memory profiling jobs are not intended for multiple subtasks per job. Modify total number of tasks or maximum number of array jobs."
            )
        if not conf["Data"]["Usage"]["evaluation_datasets"] == 1:
            raise ValueError(
                "Memoy profiling jobs are not intended for multiple evaluation datasets. Modify total number of evaluation datasets."
            )

        # for quick debugging
        logger.log(
            loglevel_info_multiple_run,
            f"Array job with {conf.param_scan_len()} executions.\n\nCONFIGURATION:\n",
        )
        logger.log(loglevel_info_multiple_run, str(conf))

        # ensure that datadir exists
        conf.make_jobscript_datadir(output_type="Memory", copy_yaml=True)
        # generate the cluster submission script
        jobscript_path = conf.generate_submission_script_from_YAML(output_type="Memory")
        print(conf["Jobscript"]["Type"], jobscript_path)

    else:  # run the actual job
        parameter_list = conf.param_scan_list()[job_idx]

        # Memory of config
        mem_dict["Memory_config"] = (
            memory_usage()[0] - mem_dict["Memory_packageoverhead"]
        )

        tmp_memory = memory_usage()[0]
        # load proper training and eval data
        evaluation_data = conf.load_evalaluation_datasets()
        training_data = conf.load_training_dataset(
            index=parameter_list[0]["training_data_index"]
        )
        trainings_index = parameter_list[0]["training_data_index"]

        # memory of data
        mem_dict["Memory_data"] = memory_usage()[0] - tmp_memory
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

            # Total memory before single run
            # mem_dict['Memory_total_beforesinglerun'] = memory_usage()[0]

            # run the single task
            mem_dict, seed = single_run(
                conf=conf,
                job_idx=job_idx,
                sub_idx=i,
                training_data=training_data,
                evaluation_data=evaluation_data,
                transient_steps=int(
                    conf["Data"]["Usage"]["tansient_length"]
                    // conf["Data"]["Creation"]["TemporalParameter"]["dt"]
                ),
                mem_dict=mem_dict,
            )
            # memory after single run
            # mem_dict['Memory_total_aftersinglerun'] = memory_usage()[0]

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
            df_parameter = pd.DataFrame.from_dict(parameter_list[i]).T
            # generate Series from valid time results
            # t_val_series = pd.Series(
            #    validsteps * conf["Data"]["Creation"]["TemporalParameter"]["dt"],
            #    name="valid_time",
            # )

            # memory before writing to output
            # mem_dict['Memory_total_beforewriting'] = memory_usage()[0]

            # write memory dict to pandas dataframe
            df_memory = pd.DataFrame(mem_dict, index=[0]).T

            # append all DataFrames to output list
            output_list.append(df_parameter)
            output_list.append(df_memory)

        # gather all DataFrames
        df_combined = pd.concat(output_list, axis=0).T
        # write output DF to file
        df_combined.to_csv(
            conf.jobscript_datadir(output_type="Memory") / f"memory-{job_idx+1}.csv",
            index=False,
        )

        logger.log(loglevel_info_multiple_run, f"Finished job {job_idx+1}")


def single_run(
    conf, job_idx, sub_idx, training_data, evaluation_data, transient_steps, mem_dict
) -> tuple[dict, int]:
    """
    Run a single task.
    That means: Evaluate a single set on parameters from the list of dictionaries returned by the Config object on a fixed number (specified in the conf object) of evaluation data files.

    """

    # get length of training/evaluation/transient data
    step_size = conf["Data"]["Creation"]["TemporalParameter"]["dt"]
    transient_steps = int(conf["Data"]["Usage"]["tansient_length"] // step_size)
    evaluation_steps = int(conf["Data"]["Usage"]["evaluation_length"] // step_size)

    # set up reservoir parameter
    tmp_mem = memory_usage()[0]
    par = ParallelReservoirsArguments.from_config(
        conf=conf, job_idx=job_idx, sub_idx=sub_idx
    )
    # memory of parameterargs
    mem_dict["Memory_ArgumentInstance"] = memory_usage()[0] - tmp_mem
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
    tmp_mem = memory_usage()[0]
    partial_train = partial(
        res.train,
        input_training_data=training_data[:-1],
        output_training_data=training_data[1 + transient_steps :],
        transient_steps=transient_steps,
    )
    mem_train = memory_usage(partial_train)
    mem_dict["Memory_train"] = np.max(mem_train) - tmp_mem

    # res.train(
    #    input_training_data=training_data[:-1],
    #    output_training_data=training_data[1 + transient_steps :],
    #    transient_steps=transient_steps,
    # )

    valid_steps = np.full(conf["Data"]["Usage"]["evaluation_datasets"], np.nan)

    logger.log(
        loglevel_info_multiple_run,
        f"Starting evaluation of job {job_idx+1} subtask {sub_idx+1}",
    )
    for evaluation_nr in range(conf["Data"]["Usage"]["evaluation_datasets"]):
        # transient update
        tmp_mem = memory_usage()[0]
        partial_transient = partial(
            res.reservoir_transient,
            input=evaluation_data[evaluation_nr][:transient_steps],
            predict_on_transient=False,
        )
        mem_transient = memory_usage(partial_transient)
        mem_dict["Memory_transient"] = np.max(mem_transient) - tmp_mem
        # res.reservoir_transient(
        #    input=evaluation_data[evaluation_nr][:transient_steps],
        #    predict_on_transient=False,
        # )

        # iteratively predict a timeseries (with consistently using 100 steps for the predictions)
        tmp_mem = memory_usage()[0]
        partial_iterative_predict = partial(
            res.iterative_predict,
            initial=evaluation_data[evaluation_nr][
                transient_steps : transient_steps + 1
            ],
            max_steps=evaluation_steps,
            supervision_data=evaluation_data[evaluation_nr][
                transient_steps + 1 : transient_steps + evaluation_steps + 1
            ],
            error_function="NRMSE",
            mean_norm=conf["Data"]["Usage"]["mean_norm"],
            error_stop=0,
            extra_steps=100,
        )
        mem_iterative_predict = memory_usage(partial_iterative_predict)
        mem_dict["Memory_predict"] = np.max(mem_iterative_predict) - tmp_mem
        # _, _, valid_steps[evaluation_nr] = res.iterative_predict(
        #    initial=evaluation_data[evaluation_nr][
        #        transient_steps : transient_steps + 1
        #    ],  # this is the first step only. Just keeping temporal dimension
        #    max_steps=evaluation_steps,
        #    supervision_data=evaluation_data[evaluation_nr][
        #        transient_steps + 1 : transient_steps + evaluation_steps + 1
        #    ],
        #    error_function="NRMSE",
        #    mean_norm=conf["Data"]["Usage"]["mean_norm"],
        #    error_stop=conf["Data"]["Usage"]["error_stop"],
        # )

        # add maximum memory usage
        mem_dict["Memory_total_max"] = np.max(
            np.concatenate((mem_train, mem_transient, mem_iterative_predict))
        )

    logger.log(
        loglevel_info_multiple_run,
        f"Finished evaluation of job {job_idx+1} subtask {sub_idx+1}",
    )
    # return the results
    return mem_dict, res.seed


if __name__ == "__main__":
    mem_dict = {}
    ## Memory overhead
    mem_dict["Memory_packageoverhead"] = memory_usage()[0]
    main(mem_dict)
