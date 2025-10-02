from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drrc.analysis import AutomaticPostprocessing
from drrc.config import Config
from drrc.tools.plot_config import *


def concatenate_csv_files(root_path: Path) -> pd.DataFrame:
    """
    Concatenates multiple CSV files into a single DataFrame.

    Args:
        root_path: root folder from which to search for DataFrame.csv

    Returns:
        pandas.DataFrame: A DataFrame resulting from the concatenation of all matched
            CSV files, or None if no files match the glob pattern.

    Raises:
        ValueError: If the glob pattern does not match any files.
    """
    # Get a list of all the csv files matching the glob pattern
    csv_files = list(root_path.glob("**/ProcessedValidTimes.csv"))

    # Concatenate all the dataframes into one
    concatenated_df = pd.concat(
        (pd.read_csv(fp) for fp in csv_files), ignore_index=True
    )

    return concatenated_df


def relative_performance(
    dfgrouped: pd.DataFrame,
    *,
    dr: float = 0.5,
) -> None:
    """Look at predictions in parallel reservoirs without dimensionality reduction"""

    # Get unique reservoir counts
    parallelreservoirs_grid_shapes = ["[1]", "[2]", "[4]", "[8]", "[16]", "[32]"]

    # get colours
    nr_plots = len(parallelreservoirs_grid_shapes)
    colormap = "viridis"
    colorrange = [0, 0.9]  # control where cmap begins (lighter darker) [0,1]
    colors = plt.get_cmap(colormap)(np.linspace(colorrange[0], colorrange[1], nr_plots))

    # for each of those, plot the performance of the above values
    fig, axs = plt.subplots(
        1, 2, figsize=figsize_single, sharey=True, constrained_layout=True
    )

    for ind, trafo in enumerate(["pca", "fft"]):
        axs[ind].set_title(trafo.upper())
        for par_ind, parres in enumerate(parallelreservoirs_grid_shapes):
            # plot identity
            id_group = dfgrouped[
                (dfgrouped["parallelreservoirs_grid_shape"] == parres)
                & (dfgrouped["Transformation"] == "identity")
                & (dfgrouped["dimensionreduction_fraction"] == 1)
            ]
            # Select data for this value1, value2 and value3
            dr_group = dfgrouped[
                (dfgrouped["parallelreservoirs_grid_shape"] == parres)
                & (dfgrouped["Transformation"] == trafo)
                & (dfgrouped["dimensionreduction_fraction"] == dr)
            ]

            # use length of shortest array:
            num = np.min(
                [
                    len(id_group["mean_ValidTime"].to_numpy()),
                    len(dr_group["mean_ValidTime"].to_numpy()),
                ]
            )

            # plotting
            axs[ind].plot(
                id_group["reservoir_nodes"].to_numpy()[:num],
                dr_group["mean_ValidTime"].to_numpy()[:num]
                / id_group["mean_ValidTime"].to_numpy()[:num],
                label=f"{parres[1:-1]}",
                color=colors[
                    par_ind
                ],  # skip first color, as 1 parallel reservoir is neglected
            )
            # plot 1
            axs[ind].axhline(1, color="black", ls="dotted")

        # set axis labels&ticks
        axs[ind].set_xlabel(labels["nodes_per_res"])
        axs[ind].set_xscale("log")
        axs[ind].set_yscale("log")

        axs[ind].set_xticks([100, 1000, 8000])
        axs[ind].set_xticklabels(["100", "1000", "8000"])

        axs[ind].set_xlim(100, 8000)
        axs[ind].set_ylim(0.1, 10)
        axs[ind].set_yticks([0.1, 1, 10])
        axs[ind].set_yticklabels(["0.1", "1", "10"])

    axs[0].set_ylabel(labels["relative_performance"])

    ## Create a shared legend for all plots
    # make room for it
    fig.suptitle(r"x\\x", color="white")  # .subplots_adjust(top=0.85)
    # plot legend
    handles, label = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        label,
        bbox_to_anchor=(0, 0.9, 1, 0.1),
        loc="upper center",
        ncol=6,
        handlelength=0.8,
        handletextpad=0.5,
        columnspacing=1,
    )
    set_figure_index(axs, x_direction=0.01, y_direction=0.99, facecol="lightgray")
    plt.savefig(
        f"{str(Config.get_git_root())}/Figures/Manuscript/RelativePerformance_{dr}.pdf",
    )
    print(
        f"Saved plot at {str(Config.get_git_root())}/Figures/Manuscript/RelativePerformance_{dr}.pdf"
    )
    plt.close()


def relative_performance_combigned(
    dfgrouped: pd.DataFrame,
) -> None:
    """Look at predictions in parallel reservoirs without dimensionality reduction"""

    # Get unique reservoir counts
    parallelreservoirs_grid_shapes = ["[1]", "[2]", "[4]", "[8]", "[16]", "[32]"]

    # get colours
    nr_plots = len(parallelreservoirs_grid_shapes)
    colormap = "viridis"
    colorrange = [0, 0.85]  # control where cmap begins (lighter darker) [0,1]
    colors = plt.get_cmap(colormap)(np.linspace(colorrange[0], colorrange[1], nr_plots))

    # for each of those, plot the performance of the above values
    fig, axs = plt.subplots(
        2,
        2,
        figsize=figsize_single_tworows,
        sharey=True,
        sharex=True,
        constrained_layout=True,
    )
    for y_ind, dr in enumerate([0.25, 0.5]):
        for x_ind, trafo in enumerate(["pca", "fft"]):
            if y_ind == 0:
                axs[y_ind, x_ind].set_title(trafo.upper())
            for par_ind, parres in enumerate(parallelreservoirs_grid_shapes):
                # plot identity
                id_group = dfgrouped[
                    (dfgrouped["parallelreservoirs_grid_shape"] == parres)
                    & (dfgrouped["Transformation"] == "identity")
                    & (dfgrouped["dimensionreduction_fraction"] == 1)
                ]
                # Select data for this value1, value2 and value3
                dr_group = dfgrouped[
                    (dfgrouped["parallelreservoirs_grid_shape"] == parres)
                    & (dfgrouped["Transformation"] == trafo)
                    & (dfgrouped["dimensionreduction_fraction"] == dr)
                ]

                # use length of shortest array:
                num = np.min(
                    [
                        len(id_group["mean_ValidTime"].to_numpy()),
                        len(dr_group["mean_ValidTime"].to_numpy()),
                    ]
                )

                # plotting
                axs[y_ind, x_ind].plot(
                    id_group["reservoir_nodes"].to_numpy()[:num],
                    dr_group["mean_ValidTime"].to_numpy()[:num]
                    / id_group["mean_ValidTime"].to_numpy()[:num],
                    label=f"{parres[1:-1]}",
                    color=colors[
                        par_ind
                    ],  # skip first color, as 1 parallel reservoir is neglected
                )
                # plot 1
                axs[y_ind, x_ind].axhline(1, color="black", ls="dotted")

            # set axis labels&ticks
            # axs[y_ind, x_ind].set_xlabel(labels['nodes_per_res'])
            axs[y_ind, x_ind].set_xscale("log")
            axs[y_ind, x_ind].set_yscale("log")

            axs[y_ind, x_ind].set_xticks([100, 1000, 8000])
            axs[y_ind, x_ind].set_xticklabels(["100", "1000", "8000"])

            axs[y_ind, x_ind].set_xlim(100, 8000)
            axs[y_ind, x_ind].set_ylim(0.1, 10)
            axs[y_ind, x_ind].set_yticks([0.1, 1, 10])
            axs[y_ind, x_ind].set_yticklabels(["0.1", "1", "10"])

            # legend
            axs[y_ind, x_ind].legend(
                [],
                [],
                title=r"$\eta$ = " + str(int(dr * 100)) + r"\%",
                loc="upper right",
                ncol=1,
                borderpad=0.2,
            )
    fig.supxlabel(labels["nodes_per_res"])
    fig.supylabel(labels["relative_performance"])

    ## Create a shared legend for all plots
    # make room for it
    fig.suptitle(r"x\\x", color="white")  # .subplots_adjust(top=0.85)
    # plot legend
    handles, label = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        label,
        bbox_to_anchor=(0, 0.9, 1, 0.1),
        loc="upper center",
        ncol=6,
        handlelength=0.8,
        handletextpad=0.5,
        columnspacing=1,
    )
    set_figure_index(
        axs.flatten(), x_direction=0.01, y_direction=0.99, facecol="lightgray"
    )
    plt.savefig(
        f"{str(Config.get_git_root())}/Figures/Manuscript/RelativePerformance_25And50.pdf",
    )
    print(
        f"Saved plot at {str(Config.get_git_root())}/Figures/Manuscript/RelativePerformance_25And50.pdf"
    )
    plt.close()


def main():
    # Load, reduce and group data
    df = concatenate_csv_files(
        Config.get_git_root() / Path("Data/ValidTimes/1D_KuramotoSivashinsky/")
    )
    df = df[
        (df["adjacency_degree"] == 2)
        & (df["input_bias"] == 0.01)
        & (df["parallelreservoirs_ghosts"] == 10)
        & (df["training_output_bias"] == 0.01)
    ]

    dfgrouped = df.loc[
        df.groupby(
            [
                "reservoir_nodes",
                "Transformation",
                "dimensionreduction_fraction",
                "parallelreservoirs_grid_shape",
            ]
        )["mean_ValidTime"].idxmax()
    ]
    # Plotting
    set_plot_style()
    dr_ratios = sorted(dfgrouped["dimensionreduction_fraction"].unique())
    for dr_ratio in dr_ratios:
        relative_performance(dfgrouped, dr=dr_ratio)
    relative_performance_combigned(dfgrouped)


if __name__ == "__main__":
    main()
