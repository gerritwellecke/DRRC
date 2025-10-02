import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drrc.config import Config
from drrc.tools.plot_config import *


def figure_TruthPrediction(
    transient_results, iterative_prediction, truth, error, error_index, filename
):  # plot the kymograph for all timesteps
    #########################################
    # if transient is plotted
    # prediction = np.append(transient_results, iterative_prediction, axis=0)
    # else, i.e. transient is not plotted
    prediction = iterative_prediction
    truth = truth[transient_results.shape[0] :]
    #########################################

    vmin, vmax = -3, 3

    figsize = tuple(np.array(figsize_single) * [1, 1.618])

    fig, axs = plt.subplots(
        3, 1, figsize=figsize, sharex=True, sharey=True, constrained_layout=True
    )
    # truth
    im = axs[0].imshow(
        truth[:, 0, :].T,
        aspect="auto",
        extent=(
            0,
            prediction.shape[0] * step_size * lyapunov_time,
            0,
            128,
        ),
        vmin=vmin,
        vmax=vmax,
        interpolation="none",
    )

    # Prediction
    im = axs[1].imshow(
        prediction[:, 0, :].T,
        aspect="auto",
        extent=(0, prediction.shape[0] * step_size * lyapunov_time, 0, 128),
        vmin=vmin,
        vmax=vmax,
        interpolation="none",
    )
    axs[2].set_xlabel(labels["time"])
    axs[1].set_ylabel(labels["space"])
    axs[0].set_yticks([0, 128])
    axs[0].set_yticklabels([0, 60])

    for ax in axs:
        # Align tick labels differently
        ax.get_yticklabels()[0].set_va("bottom")
        ax.get_yticklabels()[-1].set_va("top")

        # Change x- and y-ticks direction to 'in'
        ax.tick_params(axis="x", direction="in")
        ax.tick_params(axis="y", direction="in")

    # Absolute difference
    error_im = axs[2].imshow(
        error[:, 0, :].T,
        aspect="auto",
        extent=(0, prediction.shape[0] * step_size * lyapunov_time, 0, 128),
        vmin=vmin,
        vmax=vmax,
        interpolation="none",
    )

    for ax in axs:
        ax.vlines(
            error_index * step_size * lyapunov_time,
            0,
            128,
            color="black",
            linestyle="--",
        )
    print(error_index * step_size * lyapunov_time)

    cbar = fig.colorbar(
        error_im, ax=axs, orientation="vertical", location="right", pad=0.03
    )

    set_figure_index(axs)

    # plt.show()
    plt.savefig(f"{Config.get_git_root()}/Figures/Manuscript/{filename}.pdf")
    plt.close()
    print(
        f"Saved Figure at {Config.get_git_root()}/Figures/Manuscript/{filename}.pdf",
    )


if __name__ == "__main__":
    # load plot data
    data = np.load(
        Config.get_git_root() / Path("Data/Manuscript/TimeSeries_truth_prediction.npz")
    )
    truth = data["truth"]
    transient_results = data["transient_results"]
    prediction = data["prediction"]
    error_index = data["error_index"]

    # generate plot
    set_plot_style()
    figure_TruthPrediction(
        transient_results=transient_results,
        iterative_prediction=prediction,
        truth=truth,
        error=truth - np.append(transient_results, prediction, axis=0),
        error_index=error_index,
        filename="TimeSeries_TruthPrediction",
    )
