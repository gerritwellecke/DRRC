""" This script is only preliminary. It is used or debugging and testing the parallel reservoirs. But needs a lot of work to be nice.
"""
import logging as log
import os.path as paths
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from drrc.config import Config
from drrc.parallelreservoirs import (
    ParallelReservoirs,
    ParallelReservoirsArguments,
    ParallelReservoirsFFT,
    ParallelReservoirsPCA,
)
from drrc.tools.logger_config import drrc_logger as logger
from drrc.tools.visualization import visualization as vis

# set up logging
loglevel_info_single_run = log.getLevelName("INFO_1RUN")
logger.propagate = False
logger.setLevel(log.getLevelName("INFO_1RUN"))  # INFO_nRUN and INFO_1RUN


def visualize_1d_for_Talk(
    prediction, truth, absolute_error, error, filename
):  # plot the kymograph for all timesteps
    step_size = 0.25
    vmin, vmax = np.min(truth), np.max(truth)

    # Prediction
    fig, axs = plt.subplots(1, 3, figsize=(7, 3), dpi=1000, sharey=True, sharex=True)
    im = axs[0].imshow(
        prediction[::-1, 0, :],
        aspect="auto",
        extent=(0, 128, 0, prediction.shape[0] * step_size),
        vmin=vmin,
        vmax=vmax,
        interpolation="none",
    )
    axs[1].set_xlabel(r"space $x$")
    axs[0].set_title("Prediction")
    axs[0].set_ylabel(r"time $t$")
    axs[0].set_xticks([0, 32, 64, 96, 128])

    # truth
    im = axs[1].imshow(
        truth[::-1, 0, :],
        aspect="auto",
        extent=(0, 128, 0, prediction.shape[0] * step_size),
        vmin=vmin,
        vmax=vmax,
        interpolation="none",
    )
    axs[1].set_title("Truth")

    # Absolute difference
    error_im = axs[2].imshow(
        absolute_error[::-1, 0, :],
        aspect="auto",
        extent=(0, 128, 0, prediction.shape[0] * step_size),
        vmin=0,
        vmax=vmax,
        interpolation="none",
    )
    axs[2].set_title(r"abs. Error")

    plt.savefig(
        f"{Config.get_git_root()}/Figures/Reservoir/{filename}.png", bbox_inches="tight"
    )
    plt.close()
    logger.log(
        loglevel_info_single_run,
        f"Saved Figure at {Config.get_git_root()}/Figures/Reservoir/{filename}.png",
    )


def visualize_timeseries(
    prediction, filename, **kwargs
):  # plot the kymograph for all timesteps
    """This is just preliminary to see how the results look like"""
    step_size = 0.25

    # different plots for different dimension
    if len(prediction.shape) == 1 + 2:  # 1d data
        figs = 1
        if "truth" in kwargs:
            figs += 1
            vmin, vmax = np.min(kwargs["truth"]), np.max(kwargs["truth"])
        else:
            vmin, vmax = np.min(prediction), np.max(prediction)
        if "abs_diff" in kwargs:
            figs += 1
        if "error" in kwargs:
            figs += 1
        fig, axs = plt.subplots(
            figs, dpi=300, sharex=True, figsize=(4.5, 4.5 / 1.6 * figs)
        )
        fig.subplots_adjust(right=0.8)
        # Prediction
        im = axs[0].imshow(
            prediction[:, 0, :].T,
            aspect="auto",
            extent=(0, prediction.shape[0] * step_size, 0, 60),
            vmin=vmin,
            vmax=vmax,
            interpolation="none",
        )
        axs[0].set_ylabel(r"space $x$")
        axs[0].set_title("Prediction")
        next_plot = 1
        if "truth" in kwargs:
            im = axs[next_plot].imshow(
                kwargs["truth"][:, 0, :].T,
                aspect="auto",
                extent=(0, prediction.shape[0] * step_size, 0, 60),
                vmin=vmin,
                vmax=vmax,
                interpolation="none",
            )
            axs[next_plot].set_title("Truth")
            axs[next_plot].set_ylabel(r"space $x$")
            next_plot += 1
        # cbar:
        if "truth" in kwargs:
            c_y0 = axs[1].get_position().y0
            c_height = axs[0].get_position().y1 - axs[1].get_position().y0
        else:
            c_y0 = axs[0].get_position().y0
            c_height = axs[0].get_position().height
        cbar_ax = fig.add_axes(
            [0.85, c_y0, 0.02, c_height]
        )  # [left, bottom, width, height]
        plt.colorbar(im, cax=cbar_ax, label=r"$u(x, t)$")
        if "abs_diff" in kwargs:
            error_im = axs[next_plot].imshow(
                kwargs["abs_diff"][:, 0, :].T,
                aspect="auto",
                extent=(0, prediction.shape[0] * step_size, 0, 60),
                vmin=0,
                vmax=vmax,
                interpolation="none",
            )
            axs[next_plot].set_title("Absolute Difference")
            axs[next_plot].set_ylabel(r"space $x$")
            # cbar_ax = fig.add_axes([0.85, 0.38, 0.02, 0.5])  # [left, bottom, width, height]
            # error_cbar_ax = fig.add_axes([0.85, 0.13, 0.02, 0.2])  # [left, bottom, width, height]
            # cbar
            c_e_y0 = axs[next_plot].get_position().y0
            c_height = axs[next_plot].get_position().height
            error_cbar_ax = fig.add_axes(
                [0.85, c_e_y0, 0.02, c_height]
            )  # [left, bottom, width, height]
            plt.colorbar(
                error_im, cax=error_cbar_ax, label=r"$|u(x, t) - \hat{u}(x, t)|$"
            )
            next_plot += 1
        # Error subpanal
        if "error" in kwargs:
            error = kwargs["error"]
            axs[next_plot].plot(
                np.arange(0, step_size * len(error), step_size), error, color="red"
            )
            axs[next_plot].hlines(
                y=[eval_threshhold],
                xmin=[0],
                xmax=[step_size * len(error)],
                colors=["black"],
                linestyles="dashed",
            )
            axs[next_plot].vlines(
                x=[step_size * (len(error) - extra_steps)],
                ymin=[0],
                ymax=[np.max(error)],
                colors=["r"],
                linestyles="dashed",
            )
            axs[next_plot].set_title("Error")
        axs[-1].set_xlabel(r"time $t$")
        plt.savefig(f"{Config.get_git_root()}/Figures/Reservoir/{filename}.pdf")
        plt.close()
        logger.log(
            loglevel_info_single_run,
            f"Saved Figure at {Config.get_git_root()}/Figures/Reservoir/{filename}.pdf",
        )

    elif len(prediction.shape) == (2 + 2):  # 2d data
        frames = prediction.shape[0]
        t = np.linspace(start=0, stop=frames - 1, num=frames)
        if not "truth" in kwargs and prediction.shape[1] == 1:
            vis.produce_animation(
                t=t,
                x=128,
                dx=1,
                fps=10,
                T_out=(eval_data[:frames].shape[0] - 1) // 10,
                animation_data=[prediction[:, 0]],
                bounds=None,  # [[0,1], [0,3]],
                names=["Prediction u"],
                plt_shape=(1, 1),  # (rows, cols)
                Path=f"{Config.get_git_root()}/Figures/Reservoir/",
                figure_name=filename,
                prog_feedback=True,
                one_cbar=True,
            )
        if not "truth" in kwargs and prediction.shape[1] == 2:
            vis.produce_animation(
                t=t,
                x=128,
                dx=1,
                fps=10,
                T_out=(eval_data[:frames].shape[0] - 1) // 10,
                animation_data=[prediction[:, 0]],
                bounds=None,  # [[0,1], [0,3]],
                names=["Prediction u", "Prediction w"],
                plt_shape=(1, 1),  # (rows, cols)
                Path=f"{Config.get_git_root()}/Figures/Reservoir/",
                figure_name=filename,
                prog_feedback=True,
                one_cbar=True,
            )
        elif "truth" in kwargs and prediction.shape[1] == 1:
            vis.produce_animation(
                t=t,
                x=128,
                dx=1,
                fps=10,
                T_out=(eval_data[:frames].shape[0] - 1) // 10,
                animation_data=[kwargs["truth"][:, 0], prediction[:, 0]],
                bounds=[
                    [np.min(kwargs["truth"][:, 0]), np.max(kwargs["truth"][:, 0])],
                    [np.min(kwargs["truth"][:, 0]), np.max(kwargs["truth"][:, 0])],
                ],  # [[0,1], [0,3]],
                names=["Truth u", "Prediction u"],
                plt_shape=(1, 2),  # (rows, cols)
                Path=f"{Config.get_git_root()}/Figures/Reservoir/",
                figure_name=filename,
                prog_feedback=True,
                one_cbar=True,
            )
        elif "truth" in kwargs and prediction.shape[1] == 2:
            vis.produce_animation(
                t=t,
                x=128,
                dx=1,
                fps=10,
                T_out=(eval_data[:frames].shape[0] - 1) // 10,
                animation_data=[
                    kwargs["truth"][:, 0],
                    prediction[:, 0],
                    kwargs["truth"][:, 1],
                    prediction[:, 1],
                ],
                bounds=[
                    [np.min(kwargs["truth"][:, 0]), np.max(kwargs["truth"][:, 0])],
                    [np.min(kwargs["truth"][:, 0]), np.max(kwargs["truth"][:, 0])],
                    [np.min(kwargs["truth"][:, 1]), np.max(kwargs["truth"][:, 1])],
                    [np.min(kwargs["truth"][:, 1]), np.max(kwargs["truth"][:, 1])],
                ],  # [[0,1], [0,3]],
                names=["Truth u", "Prediction u", "Truth w", "Prediction w"],
                plt_shape=(2, 2),
                Path=f"{Config.get_git_root()}/Figures/Reservoir/",
                figure_name=filename,
                prog_feedback=True,
                one_cbar=False,
            )
        elif "truth" in kwargs and "abs_diff" in kwargs and prediction.shape[1] == 1:
            vis.produce_animation(
                t=t,
                x=128,
                dx=1,
                fps=10,
                T_out=(eval_data[:frames].shape[0] - 1) // 10,
                animation_data=[
                    kwargs["truth"][:, 0],
                    prediction[:, 0],
                    kwargs["abs_diff"][:, 0],
                ],
                bounds=None,  # [[0,1], [0,3]],
                names=["Truth u", "Prediction u", "Absolute Difference"],
                plt_shape=(1, 3),
                Path=f"{Config.get_git_root()}/Figures/Reservoir/",
                figure_name=filename,
                prog_feedback=True,
                one_cbar=True,
            )
        elif "truth" in kwargs and "abs_diff" in kwargs and prediction.shape[1] == 2:
            vis.produce_animation(
                t=t,
                x=128,
                dx=1,
                fps=10,
                T_out=(eval_data[:frames].shape[0] - 1) // 10,
                animation_data=[
                    kwargs["truth"][:, 0],
                    prediction[:, 0],
                    kwargs["abs_diff"][:, 0],
                    kwargs["truth"][:, 1],
                    prediction[:, 1],
                    kwargs["abs_diff"][:, 1],
                ],
                bounds=None,  # [[0,1], [0,3]],
                names=[
                    "Truth u",
                    "Prediction u",
                    "Absolute Difference u",
                    "Truth w",
                    "Prediction w",
                    "Absolute Difference w",
                ],
                plt_shape=(2, 3),
                Path=f"{Config.get_git_root()}/Figures/Reservoir/",
                figure_name=filename,
                prog_feedback=True,
                one_cbar=False,
            )


# All variables which need to be set to modify the data
training_steps = 80000
transient_steps = 100

evaluation_steps = 1000
mean_norm = 14.77
eval_threshhold = 0.5
extra_steps = 500


logger.log(loglevel_info_single_run, "Loading data")
# load data - 1D Kuramoto Sivashinsky
saving_path = "/1D_KuramotoSivashinsky/"
eval_data = np.load(
    str(Config.get_git_root()) + "/Data/1D_KuramotoSivashinsky/EvaluationData0.npy"
)
train_data = np.load(
    str(Config.get_git_root()) + "/Data/1D_KuramotoSivashinsky/TrainingData0.npy"
)
## add multiple variale axis - which is 1 in this case
## this here is taylored to the 1D Kuramoto Sivashinsky data, adding axis is not nesessary for aliev panfilov
eval_data = eval_data[:, np.newaxis, :]
train_data = train_data[:, np.newaxis, :]

# load data - 2D Aliev Panfilov
# saving_path = "/2D_AlievPanfilov/"
# eval_data = np.load(str(Config.get_git_root())+"/Data/2D_AlievPanfilov/EvaluationData0.npz")['vars']
# train_data = np.load(str(Config.get_git_root())+"/Data/2D_AlievPanfilov/TrainingData0.npz")['vars']

# TESTING with sinusoidal data
##eval_data = np.load("../Data/Test_TrainingData0.npy")
##train_data = np.load("../Data/Test_TrainingData0.npy")
##eval_data.fill(1)
##train_data.fill(1)

# if necessary, reducing trainingsteps to largest possible
if transient_steps + training_steps + 1 > train_data.shape[0]:
    training_steps = train_data.shape[0] - transient_steps - 1
    logger.warning(
        f"Transient steps + training steps + 1 is larger than the length of training data. Reducing the number of training steps to {training_steps} (the largest possible)."
    )
# setting up training and evaluation data
## first step of data is not used as output and last steps of data is not used as input
## transient steps are
input = train_data[: (transient_steps + training_steps)]
output = train_data[1 + transient_steps : (transient_steps + training_steps + 1)]
logger.log(loglevel_info_single_run, "Done.")


# Create a ParallelReservoir object
# TODO set parameters from yml file
KS_par = ParallelReservoirsArguments(
    adjacency_degree=2,
    adjacency_dense=False,
    adjacency_spectralradius=3.162278,
    reservoir_leakage=0.9,
    reservoir_nodes=8000,
    input_scaling=0.017783,
    input_bias=0,
    spatial_shape=(128,),
    system_variables=1,
    boundary_condition="Periodic",
    parallelreservoirs_grid_shape=(8,),
    parallelreservoirs_ghosts=10,
    dimensionreduction_fraction=0.5,
    training_includeinput=True,
    training_regularization=0.000001,
    training_output_bias=0.01,
    identical_inputmatrix=True,
    identical_adjacencymatrix=True,
    identical_outputmatrix=(0,),  # (16,16),
)

par = ParallelReservoirsArguments(
    adjacency_degree=2,
    adjacency_dense=False,
    adjacency_spectralradius=0.00999999999999999,
    reservoir_leakage=1.0,
    reservoir_nodes=5000,
    input_scaling=0.01,
    input_bias=0,  # .01,
    spatial_shape=(128,),
    system_variables=1,
    boundary_condition="Periodic",
    parallelreservoirs_grid_shape=(2,),
    parallelreservoirs_ghosts=10,
    dimensionreduction_fraction=1.0,
    training_includeinput=True,
    training_regularization=1e-06,
    training_output_bias=0.01,
    identical_inputmatrix=True,
    identical_adjacencymatrix=True,
    identical_outputmatrix=(0,),
)


AP_par = ParallelReservoirsArguments(
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

# res = ParallelReservoirs(Parameter=KS_par)
res = ParallelReservoirsPCA(Parameter=KS_par, prediction_model="1D_KuramotoSivashinsky")
# res = ParallelReservoirsPCA(Parameter=AP_par, prediction_model="2D_AlievPanfilov")
# res = ParallelReservoirsFFT(Parameter=AP_par, prediction_model="2D_AlievPanfilov")

# train the reservoir
res.train(
    input_training_data=input,
    output_training_data=output,
    transient_steps=transient_steps,
)

# transient: update the reservoir state with the evaluation data, get one step ahead predictions on transient
transient_results = res.reservoir_transient(
    input=eval_data[:transient_steps], predict_on_transient=True
)


visualize_timeseries(
    prediction=transient_results,
    truth=eval_data[1 : transient_steps + 1],
    abs_diff=np.abs(transient_results - eval_data[1 : transient_steps + 1]),
    filename=f"{saving_path}/Test_transient",
)


# iteratively predict a timeseries
results = res.iterative_predict(
    initial=eval_data[
        transient_steps : transient_steps + 1
    ],  # this is the first step only. Just keeping temporal dimension
    max_steps=evaluation_steps,
    supervision_data=eval_data[
        transient_steps + 1 : transient_steps + evaluation_steps + 1
    ],
    error_function="NRMSE",
    mean_norm=mean_norm,
    error_stop=eval_threshhold,
    extra_steps=extra_steps,
)

np.savez_compressed(
    Config.get_git_root() / Path("Data/Manuscript/TimeSeries_truth_prediction.npz"),
    truth=eval_data[1 : transient_steps + 1 + results[2] + extra_steps],
    transient_results=transient_results,
    prediction=results[0],
    error_index=results[2],
)
logger.log(
    loglevel_info_single_run,
    f"Saved predicted timeseries at {Config.get_git_root()}/Data/Manuscript/TimeSeries_truth_prediction.npz",
)


visualize_timeseries(
    prediction=transient_results,
    truth=eval_data[1 : transient_steps + 1],
    abs_diff=np.abs(transient_results - eval_data[1 : transient_steps + 1]),
    filename=f"{saving_path}/Test_transient",
)

visualize_1d_for_Talk(
    prediction=results[0],
    truth=eval_data[
        transient_steps + 1 : transient_steps + 1 + results[2] + extra_steps
    ],
    absolute_error=np.abs(
        results[0]
        - eval_data[
            transient_steps + 1 : transient_steps + 1 + results[2] + extra_steps
        ]
    ),
    error=results[1],
    filename=f"{saving_path}/Test_Talk",
)

visualize_timeseries(
    prediction=results[0],
    truth=eval_data[
        transient_steps + 1 : transient_steps + 1 + results[2] + extra_steps
    ],
    abs_diff=np.abs(
        results[0]
        - eval_data[
            transient_steps + 1 : transient_steps + 1 + results[2] + extra_steps
        ]
    ),
    filename=f"{saving_path}/Test_prediction",
    error=results[1],
)
