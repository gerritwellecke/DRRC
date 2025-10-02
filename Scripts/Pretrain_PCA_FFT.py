""" This script is created to pretrain all PCA objects and select all FFT modes for the parallel reservoirs given in a specific parameterfile.
Independent of the used transformation, the PCA object and the FFT mode are trained and selected for all combinations of parallelreservoirs_grid_shape and parallelreservoirs_ghosts (and system variables) given in the parameterfile.

.. codeauthor:: Luk Fleddermann
"""

import argparse
import logging as log
from pathlib import Path

from drrc.config import Config
from drrc.parallelreservoirs import (
    ParallelReservoirsArguments,
    ParallelReservoirsFFT,
    ParallelReservoirsPCA,
)
from drrc.tools.logger_config import drrc_logger as logger

# set up logging
loglevel_info_multiple_run = log.getLevelName("INFO_nRUN")
logger.propagate = False
logger.setLevel(log.getLevelName("INFO_nRUN"))  # INFO_nRUN and INFO_1RUN


def Pretrain_1D_KuramotoSivashinsky(Pardicts: list[dict]):
    """
    Pretrain PCA and FFT for 1D for all combinations of :code:`parallelreservoirs_grid_shape` and :code:`parallelreservoirs_ghosts` given in the input list of parameter dictionaries :code:`Pardicts`.

    Args:
        Pardicts: List of dictionaries containing the parameters for the ParallelReservoirsArguments object. The dictionaries must contain the keys :code:`parallelreservoirs_grid_shape` and :code:`parallelreservoirs_ghosts`.

    Note:
        Only the dimensionallity of the input, determained by :code:`spatial_shape`, :code:`system_variables`, :code:`parallelreservoirs_grid_shape` and :code:`parallelreservoirs_ghosts`, is relevant for the training of the PCA object and FFT mode selection.
        For 1D Kuramoto_Sivashinsky, :code:`system_variables=1` and :code:`spatial_shape=128` is fixed.
    """

    # Initialize a ParallelReservoirsArguments object
    # All parameters are irrelevant.
    par = ParallelReservoirsArguments(
        adjacency_degree=2,
        adjacency_dense=False,
        adjacency_spectralradius=0.178,
        reservoir_leakage=1,
        reservoir_nodes=10,
        input_scaling=0.031623,
        input_bias=1,
        spatial_shape=(128,),
        system_variables=1,
        boundary_condition="Periodic",
        parallelreservoirs_grid_shape=(1,),
        parallelreservoirs_ghosts=10,
        dimensionreduction_fraction=1,
        training_includeinput=False,
        training_regularization=0.000001,
        training_output_bias=0.01,
        identical_inputmatrix=True,
        identical_adjacencymatrix=True,
        identical_outputmatrix=(1,),
    )

    for i, parameter in enumerate(Pardicts):
        if (
            (i == 0)
            or (
                par.parallelreservoirs_grid_shape
                != parameter["parallelreservoirs_grid_shape"]
            )
            or (par.parallelreservoirs_ghosts != parameter["parallelreservoirs_ghosts"])
        ):
            # Update relevant parameters
            par.parallelreservoirs_grid_shape = parameter[
                "parallelreservoirs_grid_shape"
            ]
            par.parallelreservoirs_ghosts = parameter["parallelreservoirs_ghosts"]

            # Initialize the ParallelReservoirsPCA object
            # This trains and saves the PCA object, if it does not exist oalready
            res = ParallelReservoirsPCA(
                Parameter=par, prediction_model="1D_KuramotoSivashinsky"
            )

            # Initialize the ParallelReservoirsFFT object
            # This selects and saves the FFT mode, if they dont already exist
            res = ParallelReservoirsFFT(
                Parameter=par, prediction_model="1D_KuramotoSivashinsky"
            )


def Pretrain_2D_AlievPanfilov(Pardicts: list[dict]):
    """
    Pretrain PCA and FFT for 1D for all combinations of :code:`parallelreservoirs_grid_shape` and :code:`parallelreservoirs_ghosts` given in the input list of parameter dictionaries :code:`Pardicts`.

    Args:
        Pardicts: List of dictionaries containing the parameters for the ParallelReservoirsArguments object. The dictionaries must contain the keys :code:`parallelreservoirs_grid_shape` and :code:`parallelreservoirs_ghosts`.

    Note:
        Only the dimensionallity of the input, determained by :code:`spatial_shape`, :code:`system_variables`, :code:`parallelreservoirs_grid_shape` and :code:`parallelreservoirs_ghosts`, is relevant for the training of the PCA object and FFT mode selection.
        For 2D Aliev Panfilov :code:`spatial_shape=(128, 128)` is fixed.
    """

    # Initialize a ParallelReservoirsArguments object
    # All parameters are irrelevant, up to spatial_shape,system_variables,parallelreservoirs_grid_shape, parallelreservoirs_ghosts.
    par = ParallelReservoirsArguments(
        adjacency_degree=2,
        adjacency_dense=False,
        adjacency_spectralradius=0.178,
        reservoir_leakage=1,
        reservoir_nodes=10,
        input_scaling=0.031623,
        input_bias=0.01,
        spatial_shape=(128, 128),
        system_variables=1,
        boundary_condition="NoFlux",
        parallelreservoirs_grid_shape=(1, 1),
        parallelreservoirs_ghosts=5,
        dimensionreduction_fraction=1,
        training_includeinput=False,
        training_regularization=0.000001,
        training_output_bias=0.01,
        identical_inputmatrix=True,
        identical_adjacencymatrix=True,
        identical_outputmatrix=(16, 16),
    )

    for i, parameter in enumerate(Pardicts):
        if (
            (i == 0)
            or (
                par.parallelreservoirs_grid_shape
                != parameter["parallelreservoirs_grid_shape"]
            )
            or (par.parallelreservoirs_ghosts != parameter["parallelreservoirs_ghosts"])
        ):
            # Update relevant parameters
            par.parallelreservoirs_grid_shape = parameter[
                "parallelreservoirs_grid_shape"
            ]
            par.parallelreservoirs_ghosts = parameter["parallelreservoirs_ghosts"]
            par.system_variables = parameter["system_variables"]

            # Initialize the ParallelReservoirsPCA object
            # This trains and saves the PCA object, if it does not exist oalready
            res = ParallelReservoirsPCA(
                Parameter=par, prediction_model="2D_AlievPanfilov"
            )

            # Initialize the ParallelReservoirsFFT object
            # This selects and saves the FFT mode, if they dont already exist
            res = ParallelReservoirsFFT(
                Parameter=par, prediction_model="2D_AlievPanfilov"
            )


def main():
    # argparser, get the index of simulation & yaml file path
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml", type=str, default=None)
    args = parser.parse_args()

    # load configuration
    conf = Config(Path(args.yaml).absolute())

    # get and flatten the parameter list
    parameter_list = conf.param_scan_list()
    parameter_list = [
        dictionary for sublist in parameter_list for dictionary in sublist
    ]

    # Train all PCA objects and select all FFT modes for the given model
    if (
        f"{conf['Data']['model_dimension']}D_{conf['Data']['model_name']}"
        == "1D_KuramotoSivashinsky"
    ):
        Pretrain_1D_KuramotoSivashinsky(parameter_list)
    elif (
        f"{conf['Data']['model_dimension']}D_{conf['Data']['model_name']}"
        == "2D_AlievPanfilov"
    ):
        Pretrain_2D_AlievPanfilov(parameter_list)


if __name__ == "__main__":
    main()
