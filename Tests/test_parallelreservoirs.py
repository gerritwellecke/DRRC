import itertools
import logging as log
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from drrc.config import Config
from drrc.parallelreservoirs import (
    ParallelReservoirs,
    ParallelReservoirsArguments,
    ParallelReservoirsFFT,
    ParallelReservoirsPCA,
)
from drrc.reservoircomputer import ReservoirComputer

# load the test yaml
# @pytest.fixture
# def conf():
#    return Config("Tests/testdata/test.yml")
#
#
## load the test yaml without parameter scan values
# @pytest.fixture
# def conf_noscan():
#    return Config("Tests/testdata/test-no_scan.yml")


def test_initialization_of_parallelreservoirs() -> None:
    """Test the initialization of the ParallelReservoirsBase class."""
    # TODO: Set parameters to good ones from known runs
    # TODO: test multiple initializations, especially all the edge cases
    par = ParallelReservoirsArguments(
        adjacency_degree=2,
        adjacency_dense=False,
        adjacency_spectralradius=0.178,
        reservoir_leakage=1,
        reservoir_nodes=10,
        input_scaling=0.031623,
        input_bias=1,
        spatial_shape=(128,),  # (128,),
        system_variables=2,
        boundary_condition="Periodic",
        parallelreservoirs_grid_shape=(4,),  # (128,),
        parallelreservoirs_ghosts=10,
        dimensionreduction_fraction=1,
        training_includeinput=False,
        training_regularization=0.000001,
        training_output_bias=0.01,
        identical_inputmatrix=True,
        identical_adjacencymatrix=True,  # TODO: I think if identical_outputmatrix is true this should default to true
        identical_outputmatrix=(1,),
    )

    # test _initialize_reservoirs
    res = ParallelReservoirs(Parameter=par)
    assert len(res.reservoirs) == np.prod(res.rc_grid_shape)
    assert type(res.reservoirs[0]) == ReservoirComputer

    # test _initialize_reservoir_slices and prediction_slice in multiple dimensions
    spatial_shapes = [(128,), (128, 128), (128, 128, 128)]
    res_grids = [(2,), (2, 5), (2, 5, 3)]
    for spatial_shape, grid in zip(spatial_shapes, res_grids):
        par.spatial_shape = spatial_shape
        par.parallelreservoirs_grid_shape = grid
        res = ParallelReservoirs(Parameter=par)
        assert len(res.reservoir_slices) == np.prod(res.rc_grid_shape)
        data = np.empty((100, 2, *res.spatial_shape))
        true_shape = (100, 2) + tuple(
            np.array(res.spatial_shape) // np.array(res.rc_grid_shape)
            + 2 * res.rc_ghosts
        )
        assert np.all(
            data[res.reservoir_slices[0]].shape == true_shape
        ), f"Slice objects for 3d data is wrong. It is {data[res.reservoir_slices[0]].shape} but needs to be {true_shape}"
        true_shape = (100, 2) + tuple(
            np.array(res.spatial_shape) // np.array(res.rc_grid_shape)
        )
        assert np.all(
            data[res.reservoir_slices[0]][res.remove_ghosts].shape == true_shape
        ), f"Prediction slice is wrong. It is {data[res.remove_ghosts].shape} but needs to be {true_shape}"


def test_boundary_condition() -> None:
    """Test the _boundary_condition method of the ParallelReservoirsBase class."""
    # Create an instance of ParallelReservoirsBase
    par = ParallelReservoirsArguments(
        adjacency_degree=4,
        adjacency_dense=False,
        adjacency_spectralradius=0.178,
        reservoir_leakage=1,
        reservoir_nodes=10,
        input_scaling=0.031623,
        input_bias=1,
        spatial_shape=(128, 128),  # (128,),
        system_variables=1,
        boundary_condition="Periodic",
        parallelreservoirs_grid_shape=(2, 5),  # (128,),
        parallelreservoirs_ghosts=2,
        dimensionreduction_fraction=1,
        training_includeinput=False,
        training_regularization=0.0001,
        training_output_bias=1,
        identical_inputmatrix=False,
        identical_adjacencymatrix=False,
        identical_outputmatrix=False,
    )

    res = ParallelReservoirs(Parameter=par)
    # Define 1d test data
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype="float").reshape(1, 1, 9)
    # Test with periodic boundary condition and add_ghostcells=False
    expected_result = np.array([6, 7, 3, 4, 5, 6, 7, 3, 4], dtype="float").reshape(
        1, 1, 9
    )
    result = res._boundary_condition(data, add_ghostcells=False)
    assert np.array_equal(result, expected_result)

    # Define 2d test data
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype="float").reshape(
        1, 1, 3, 3
    )
    # Test with periodic boundary condition and add_ghostcells=True
    # modify ghost cells
    par.parallelreservoirs_ghosts = 1
    res = ParallelReservoirs(Parameter=par)
    expected_result = np.array(
        [
            [9, 7, 8, 9, 7],
            [3, 1, 2, 3, 1],
            [6, 4, 5, 6, 4],
            [9, 7, 8, 9, 7],
            [3, 1, 2, 3, 1],
        ],
        dtype="float",
    ).reshape(1, 1, 5, 5)
    result = res._boundary_condition(data, add_ghostcells=True)
    assert np.array_equal(result, expected_result)

    # Test with no flux boundary condition and add_ghostcells=True
    res.boundary_condition = "NoFlux"
    expected_result = np.array(
        [
            [1, 1, 2, 3, 3],
            [1, 1, 2, 3, 3],
            [4, 4, 5, 6, 6],
            [7, 7, 8, 9, 9],
            [7, 7, 8, 9, 9],
        ],
        dtype="float",
    ).reshape(1, 1, 5, 5)
    result = res._boundary_condition(data, add_ghostcells=True)
    assert np.array_equal(result, expected_result)

    # Define 3d test data

    data = np.array(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype="float"
    ).reshape(1, 1, 3, 2, 2)
    # Test with periodic boundary condition and add_ghostcells=True
    par.boundary_condition = "Periodic"
    par.spatial_shape = (3, 2, 2)
    par.parallelreservoirs_grid_shape = (1, 1, 1)
    res = ParallelReservoirs(Parameter=par)
    result = res._boundary_condition(data, add_ghostcells=True)
    expected_result = np.array(
        [
            [[12, 11, 12, 11], [10, 9, 10, 9], [12, 11, 12, 11], [10, 9, 10, 9]],
            [[4, 3, 4, 3], [2, 1, 2, 1], [4, 3, 4, 3], [2, 1, 2, 1]],
            [[8, 7, 8, 7], [6, 5, 6, 5], [8, 7, 8, 7], [6, 5, 6, 5]],
            [[12, 11, 12, 11], [10, 9, 10, 9], [12, 11, 12, 11], [10, 9, 10, 9]],
            [[4, 3, 4, 3], [2, 1, 2, 1], [4, 3, 4, 3], [2, 1, 2, 1]],
        ],
        dtype="float",
    ).reshape(1, 1, 5, 4, 4)
    assert np.array_equal(result, expected_result)


def test_evaluate_prediction() -> None:
    # Create test 1d test data:
    supervision = np.zeros((1, 1, 100))
    prediction = np.ones((1, 1, 100))
    mean_norm = 20
    expected_error = np.sqrt(100) / mean_norm
    threshold = expected_error - 0.1

    truth, error = ParallelReservoirs._evaluate_one_step(
        prediction=prediction,
        supervision_data=supervision,
        errrofunction="NRMSE",
        mean_norm=mean_norm,
        error_stop=threshold,
    )
    assert truth == False
    assert error == expected_error

    # Create test 2d test data:
    supervision = np.zeros((1, 1, 100, 100))
    prediction = np.ones((1, 1, 100, 100))
    mean_norm = 20
    expected_error = np.sqrt(10000) / mean_norm

    truth, error = ParallelReservoirs._evaluate_one_step(
        prediction=prediction,
        supervision_data=supervision,
        errrofunction="NRMSE",
        mean_norm=mean_norm,
        error_stop=threshold,
    )
    assert truth == False
    assert error == expected_error


if __name__ == "__main__":
    # Define custom levels (DRRC is deeper because it is called in loops)
    RC_log_debuglevel = 13
    DRRC_log_debuglevel = 16
    ParallelReservoirs_log_debuglevel = 19

    log.root.handlers = (
        []
    )  # TODO: create handlers that use name of depth, include function...
    log.basicConfig(level=DRRC_log_debuglevel)
    test_initialization_of_parallelreservoirs()
    test_boundary_condition()
    test_evaluate_prediction()
    log.info("All tests passed.")
