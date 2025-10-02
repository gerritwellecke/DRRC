from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drrc.config import Config
from drrc.parallelreservoirs import (
    ParallelReservoirsArguments,
    ParallelReservoirsFFT,
    ParallelReservoirsPCA,
)
from drrc.tools.plot_config import *


def figure_dimensionality_reduction(grid_shapes: list[int] = [1, 4]) -> None:
    col1 = "gray"
    col2 = "tab:blue"

    res_fft = []
    res_pca = []

    for grid_shape in grid_shapes:
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
            parallelreservoirs_grid_shape=(grid_shape,),
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
        res_fft.append(
            ParallelReservoirsFFT(
                Parameter=par, prediction_model="1D_KuramotoSivashinsky"
            )
        )
        res_pca.append(
            ParallelReservoirsPCA(
                Parameter=par, prediction_model="1D_KuramotoSivashinsky"
            )
        )

    # the following is independent of res_fft and res_pca
    data = res_fft[0]._boundary_condition(
        np.load(
            Config.get_git_root()
            / Path("Data/1D_KuramotoSivashinsky/TrainingData0.npy")
        )[:, np.newaxis, :],
        add_ghostcells=True,
    )
    # from now on its not

    # Create a figure
    fig = plt.figure(layout="constrained", figsize=figsize_double)
    # Create two subfigures
    # gs = gridspec.GridSpec(1, 2, figure=fig)
    # Create two subplots in each subfigure
    rows = len(grid_shapes)
    figs = fig.subfigures(1, rows)
    axs = [
        f.subplots(2, 1)
        for f in figs
        # [fig.add_subplot(gs[0, i].subgridspec(2, 1)[j, 0]) for j in range(2)]
        # for i in range(2)
    ]

    axs[1][0].set_title("FFT")
    axs[0][0].set_title("PCA")

    axs2 = np.copy(axs)

    for i, ax_fig in enumerate(axs):
        for j, ax_plt in enumerate(ax_fig):
            components = res_fft[j]._get_input_length()

            axs2[i][j] = ax_plt.twinx()
            # for p_ind, percent in enumerate([0.25, 0.5, 0.75, 1]):
            #     axs2[j, i].vlines(
            #         [(components - 1) * percent],
            #         ymin=0,
            #         ymax=1,
            #         label=f"{percent*100}%",
            #         linestyles="dashed",
            #         colors=["red", "orange", "green", "black"][p_ind],
            #     )
            #     axs2[j, i].text(
            #         (components - 1) * (percent - 0.01),
            #         1 / 2,
            #         f"{int(percent*100)}%",
            #         ha="right",
            #         va="top",
            #         color=["red", "orange", "green", "black"][p_ind],
            #     )

            axs2[i][j].set_ylim(0, 1.1)
            axs2[i][j].set_yticks([0, 1])
            if i == 0:  # PCA
                # print(res_pca[j].pca.explained_variance_.shape)
                axs[i][j].hist(
                    np.arange(components),
                    weights=res_pca[j].pca.explained_variance_,
                    bins=components,
                    color=col1,
                )

                exp_var = res_pca[j].pca.explained_variance_
                exp_var = exp_var.cumsum() / exp_var.sum()
                xs = np.arange(components).repeat(2)[:-1]
                ys = np.zeros_like(xs)
                ys = exp_var.repeat(2)[1:]
                ys[0] = 0
                axs2[i][j].plot(xs, ys, c=col2, linewidth=1)
                if j == rows - 1:
                    axs[i][j].set_xlabel("principal component index")
            else:  # FFT
                fft_data = res_fft[j]._transform_data(
                    data[:, :, : res_fft[j]._get_input_length()], fraction=1
                )
                # print(np.max(fft_data, axis=0)[res_fft[j].largest_modes].shape)

                axs[i][j].hist(
                    np.arange(components),
                    weights=np.max(fft_data, axis=0),
                    bins=components,
                    color=col1,
                )
                exp_var = np.max(fft_data, axis=0)
                exp_var = exp_var.cumsum() / exp_var.sum()
                xs = np.arange(components).repeat(2)[:-1]
                ys = np.zeros_like(xs)
                ys = exp_var.repeat(2)[1:]
                ys[0] = 0
                axs2[i][j].plot(xs, ys, c=col2, linewidth=1)
                if j == rows - 1:
                    axs[i][j].set_xlabel("ordered mode index")
    # figs[0].supylabel(, color=col1)
    figs[0].text(
        0.02,
        0.5,
        "explained variance",
        rotation="vertical",
        color=col1,
        va="center",
        ha="center",
    )
    figs[0].text(
        0.99,
        0.5,
        r"cumulative expl. var. [\%]",
        rotation="vertical",
        color=col2,
        va="center",
        ha="center",
    )
    # figs[1].supylabel("mode amplitude", color=col1)
    figs[1].text(
        0.01,
        0.5,
        "mode amplitude",
        rotation="vertical",
        color=col1,
        va="center",
        ha="center",
    )
    figs[1].text(
        0.98,
        0.5,
        r"cumulative mode amp. [\%]",
        rotation="vertical",
        color=col2,
        va="center",
        ha="center",
    )

    axs[0][1].set_ylabel(r"X", color="white")
    axs2[0][1].set_ylabel(r"X", color="white")
    axs[1][1].set_ylabel(r"X", color="white")
    axs2[1][1].set_ylabel(r"X", color="white")

    set_figure_index(np.array(axs).flatten(), facecol="lightgray")

    plt.savefig(f"../Figures/Manuscript/Dimensionality-Reduction-{grid_shapes[-1]}.pdf")


if __name__ == "__main__":
    set_plot_style()

    figure_dimensionality_reduction()
    figure_dimensionality_reduction([1, 8])
