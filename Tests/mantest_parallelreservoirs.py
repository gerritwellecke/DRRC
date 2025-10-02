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
from drrc.tools.logger_config import drrc_logger as logger

# set up logging
loglevel_info_multiple_run = log.getLevelName("INFO_nRUN")
logger.propagate = False
logger.setLevel(log.getLevelName("INFO_nRUN"))  # INFO_nRUN and INFO_1RUN


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
        reservoir_nodes=4000,
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


def test_1d_pca_training() -> None:
    par = ParallelReservoirsArguments(
        adjacency_degree=2,
        adjacency_dense=False,
        adjacency_spectralradius=0.178,
        reservoir_leakage=1,
        reservoir_nodes=4000,
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

    # Test initialization for different PCA models
    # 1 domain
    res = ParallelReservoirsPCA(
        Parameter=par, prediction_model="1D_KuramotoSivashinsky"
    )
    # 2 domains
    par.parallelreservoirs_grid_shape = (2,)
    res = ParallelReservoirsPCA(
        Parameter=par, prediction_model="1D_KuramotoSivashinsky"
    )
    # 4 domains
    par.parallelreservoirs_grid_shape = (4,)
    res = ParallelReservoirsPCA(
        Parameter=par, prediction_model="1D_KuramotoSivashinsky"
    )
    # 8 domains
    par.parallelreservoirs_grid_shape = (8,)
    res = ParallelReservoirsPCA(
        Parameter=par, prediction_model="1D_KuramotoSivashinsky"
    )
    # 16 domains
    par.parallelreservoirs_grid_shape = (16,)
    res = ParallelReservoirsPCA(
        Parameter=par, prediction_model="1D_KuramotoSivashinsky"
    )

    ## With window size 10, the following is too much memory for a camulus node! I.e. stacking 50 trainingdatasets and 32 a slice of 24 inputs
    # 32 domains
    par.parallelreservoirs_grid_shape = (32,)
    res = ParallelReservoirsPCA(
        Parameter=par, prediction_model="1D_KuramotoSivashinsky"
    )
    ## 64 domains
    par.parallelreservoirs_grid_shape = (64,)
    res = ParallelReservoirsPCA(
        Parameter=par, prediction_model="1D_KuramotoSivashinsky"
    )
    ## 128 domains
    par.parallelreservoirs_grid_shape = (128,)
    res = ParallelReservoirsPCA(
        Parameter=par, prediction_model="1D_KuramotoSivashinsky"
    )


def test_2d_pca_training() -> None:
    # 2D AlievPanfilov
    par = ParallelReservoirsArguments(
        adjacency_degree=2,
        adjacency_dense=False,
        adjacency_spectralradius=0.178,
        reservoir_leakage=1,
        reservoir_nodes=100,
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

    # Test initialization for different PCA models
    # 1x1 domain
    res = ParallelReservoirsPCA(Parameter=par, prediction_model="2D_AlievPanfilov")
    # 2x2 domains
    par.parallelreservoirs_grid_shape = (2, 2)
    res = ParallelReservoirsPCA(Parameter=par, prediction_model="2D_AlievPanfilov")
    # 4x4 domains
    par.parallelreservoirs_grid_shape = (4, 4)
    res = ParallelReservoirsPCA(Parameter=par, prediction_model="2D_AlievPanfilov")
    # 8x8 domains
    par.parallelreservoirs_grid_shape = (8, 8)
    res = ParallelReservoirsPCA(Parameter=par, prediction_model="2D_AlievPanfilov")
    # 16x16 domains
    par.parallelreservoirs_grid_shape = (16, 16)
    res = ParallelReservoirsPCA(Parameter=par, prediction_model="2D_AlievPanfilov")
    # 32x32 domains
    par.parallelreservoirs_grid_shape = (32, 32)
    res = ParallelReservoirsPCA(Parameter=par, prediction_model="2D_AlievPanfilov")
    # 64x64 domains
    par.parallelreservoirs_grid_shape = (64, 64)
    res = ParallelReservoirsPCA(Parameter=par, prediction_model="2D_AlievPanfilov")
    # 128x128 domains
    par.parallelreservoirs_grid_shape = (128, 128)
    res = ParallelReservoirsPCA(Parameter=par, prediction_model="2D_AlievPanfilov")


def test_1d_fft() -> None:
    """Test:
    - :code:`__init__` of the ParallelReservoirsFFT class
    - :code:`_get_input_length` for one setting only (fraction=1)
    - :code:`_transform_data` by checking if the largest modes are actually the largest ones
    - :code:`_inv_transform_data` by checking if the inverse is the inverse of the transform
    """
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
    # Test initialization for different FFT models
    res = ParallelReservoirsFFT(
        Parameter=par, prediction_model="1D_KuramotoSivashinsky"
    )

    # Test _get_input_length
    assert res._get_input_length() == res.dr_fraction * np.prod(
        np.append(
            [res.system_variables],
            np.array(res.spatial_shape) // np.array(res.rc_grid_shape)
            + 2 * res.rc_ghosts,
        )
    )

    # Test fft and inverse fft
    data = res._boundary_condition(
        np.load(
            Config.get_git_root()
            / Path("Data/1D_KuramotoSivashinsky/TrainingData0.npy")
        )[:, np.newaxis, :],
        add_ghostcells=True,
    )[:, :, : res._get_input_length()]

    fft_data = np.concatenate(
        [
            np.fft.rfft(data, axis=-1).real,
            np.fft.rfft(data, axis=-1).imag[
                ..., 1 : np.ceil(data.shape[-1] / 2).astype(int)
            ],
        ],
        axis=-1,
    )
    rc_fft_data = res._transform_data(data, fraction=1)
    inv_rc_fft_data = res._inv_transform_data(rc_fft_data)

    # Test if chosen modes are actually the largest ones
    # Testing 10 maximum nodes doesnt work
    for i in range(1, 1000):  # test 1000 different time steps, this is arbitrary
        if np.max(fft_data[i, 0]) != np.max(fft_data[i, 0, res.largest_modes[:50]]):
            raise ValueError(
                f"Largest modes are not the largest ones, at index {i}, largest value of first 50 largest modes is {np.max(fft_data[i,0, res.largest_modes[:10]])} and largest mode in fft_data is {np.max(fft_data[i,0])}, but should be the same."
            )

    # Test if the inverse is inverse of transform
    if not np.allclose(data[:100], inv_rc_fft_data[:100], atol=1e-5):
        raise ValueError(
            "Transform and Inverse transform are not inverse of each other."
        )

    # Plotting results
    # Plotting maximal modes
    name = "fft_comparison"
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(
        np.max(fft_data, axis=0)[0],
        label="temporal max of FFT Data (unordered)",
        color="blue",
    )
    ax[0].plot(
        np.max(fft_data, axis=0)[0, res.largest_modes],
        label="temporal max of FFT Data (ordered)",
        color="red",
    )
    ax[1].plot(
        fft_data[0, 0], label="One timestep FFT Data (unordered)", color="tab:blue"
    )
    ax[1].plot(
        rc_fft_data[0], label="One timestep FFT Data (ordered)", color="tab:orange"
    )
    ax[2].plot(fft_data[100, 0], color="tab:blue")
    ax[2].scatter(
        res.largest_modes[: int(res.res_output_length * 0.5)],
        res._transform_data(data[100:101], fraction=0.5)[0],
        c="black",
        s=5,
        label="largest 50%",
    )
    handles, labels = [], []
    for a in ax:
        h, l = a.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=2)
    plt.savefig(
        f"{Config.get_git_root()}/Tests/TestResults/Test_{name}_{res.res_domain_size}.png",
        bbox_inches="tight",
        dpi=1000,
    )
    logger.log(
        loglevel_info_multiple_run,
        f"Saved plot to /Tests/TestResults/Test_{name}_{res.res_domain_size}.png",
    )
    plt.close()


def test_2d_fft() -> None:
    par = ParallelReservoirsArguments(
        adjacency_degree=2,
        adjacency_dense=False,
        adjacency_spectralradius=0.178,
        reservoir_leakage=1,
        reservoir_nodes=4000,
        input_scaling=0.031623,
        input_bias=0.01,
        spatial_shape=(128, 128),
        system_variables=2,
        boundary_condition="NoFlux",
        parallelreservoirs_grid_shape=(16, 16),
        parallelreservoirs_ghosts=10,
        dimensionreduction_fraction=0.25,
        training_includeinput=False,
        training_regularization=0.000001,
        training_output_bias=0.01,
        identical_inputmatrix=True,
        identical_adjacencymatrix=True,
        identical_outputmatrix=(0, 1),
    )

    # Test initialization for different FFT models
    res = ParallelReservoirsFFT(Parameter=par, prediction_model="2D_AlievPanfilov")

    # Test fft and inverse fft
    data = res._boundary_condition(
        np.load(
            Config.get_git_root() / Path("Data/2D_AlievPanfilov/TrainingData0.npz")
        )["vars"],
        add_ghostcells=True,
    )

    fft_data = np.concatenate(
        [
            np.fft.rfft2(data, axes=res.spatial_axes).real,
            np.fft.rfft2(data, axes=res.spatial_axes).imag[
                ..., 1 : np.ceil(data.shape[-1] / 2).astype(int)
            ],
        ],
        axis=-1,
    )
    rc_fft_data = res._transform_data(data, fraction=1)
    inv_rc_fft_data = res._inv_transform_data(rc_fft_data)

    # Test if chosen modes are actually the largest ones
    # Testing 10 maximum nodes doesnt work
    for i in range(1, 1000):  # test 1000 different time steps, this is arbitrary
        if np.max(fft_data[i, 0]) != np.max(fft_data[i, 0, res.largest_modes[:50]]):
            raise ValueError(
                f"Largest modes are not the largest ones, at index {i}, largest value of first 50 largest modes is {np.max(fft_data[i,0, res.largest_modes[:10]])} and largest mode in fft_data is {np.max(fft_data[i,0])}, but should be the same."
            )

    # Test if the inverse is inverse of transform
    if not np.allclose(data[:100], inv_rc_fft_data[:100], atol=1e-5):
        raise ValueError(
            "Transform and Inverse transform are not inverse of each other."
        )

    # Plotting results
    # Plotting maximal modes
    name = "2d_fft_comparison"
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(
        np.max(fft_data, axis=0)[0],
        label="temporal max of FFT Data (unordered)",
        color="blue",
    )
    ax[0].plot(
        np.max(fft_data, axis=0)[0, res.largest_modes],
        label="temporal max of FFT Data (ordered)",
        color="red",
    )
    ax[1].plot(
        fft_data[0, 0], label="One timestep FFT Data (unordered)", color="tab:blue"
    )
    ax[1].plot(
        rc_fft_data[0], label="One timestep FFT Data (ordered)", color="tab:orange"
    )
    ax[2].plot(fft_data[100, 0], color="tab:blue")
    ax[2].scatter(
        res.largest_modes[: int(res.res_output_length * 0.5)],
        res._transform_data(data[99:100], fraction=0.5)[0],
        c="black",
        s=5,
        label="largest 50%",
    )
    handles, labels = [], []
    for a in ax:
        h, l = a.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=2)
    plt.savefig(
        f"{Config.get_git_root()}/Tests/TestResults/Test_{name}.png",
        bbox_inches="tight",
        dpi=1000,
    )
    plt.close()


if __name__ == "__main__":
    # Define custom levels (DRRC is deeper because it is called in loops)

    #    test_initialization_of_parallelreservoirs()
    #    test_boundary_condition()
    #    test_evaluate_prediction()
    #    test_1d_pca_training()
    #    test_2d_pca_training()
    test_1d_fft()
    # test_2d_fft()
