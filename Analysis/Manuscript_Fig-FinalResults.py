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
    # print(*sorted(csv_files), sep="\n")

    # Concatenate all the dataframes into one
    concatenated_df = pd.concat(
        (pd.read_csv(fp) for fp in csv_files), ignore_index=True
    )

    return concatenated_df


def dr_performance(
    dfgrouped: pd.DataFrame,
    *,
    parres: str = "[4]",
) -> None:
    """Look at predictions in parallel reservoirs without dimensionality reduction"""

    # Get unique reservoir counts
    parallelreservoirs_grid_shape = parres
    dr_ratio = sorted(dfgrouped["dimensionreduction_fraction"].unique())

    # get colours
    nr_plots = len(dr_ratio)
    colormap = "magma"
    colorrange = [0.4, 0.8]  # control where cmap begins (lighter darker) [0,1]
    colors = plt.get_cmap(colormap)(np.linspace(*colorrange, nr_plots))

    # for each of those, plot the performance of the above values
    fig, axs = plt.subplots(
        1, 2, figsize=figsize_single, sharey=True, constrained_layout=True
    )

    # plot identity
    group = dfgrouped[
        (dfgrouped["parallelreservoirs_grid_shape"] == parres)
        & (dfgrouped["Transformation"] == "identity")
        & (dfgrouped["dimensionreduction_fraction"] == 1.0)
    ]
    for ax in axs:
        ax.plot(
            group["reservoir_nodes"],
            group["mean_ValidTime"] * lyapunov_time,
            color="black",
            ls="dotted",
            label="Identity",
        )

    for ind, trafo in enumerate(["pca", "fft"]):
        axs[ind].set_title(trafo.upper())
        for dr_ind, r in enumerate(dr_ratio[::-1]):
            # Select data for this value1, value2 and value3
            group = dfgrouped[
                (
                    dfgrouped["parallelreservoirs_grid_shape"]
                    == parallelreservoirs_grid_shape
                )
                & (dfgrouped["Transformation"] == trafo)
                & (dfgrouped["dimensionreduction_fraction"] == r)
            ]
            axs[ind].plot(
                group["reservoir_nodes"],
                group["mean_ValidTime"] * lyapunov_time,
                label=f"{int(r*100)} " + r"\%",
                color=colors[dr_ind],
            )
            # axs[ind].plot(
            #    group["reservoir_nodes"],
            #    group["input_scaling"],
            #    label=f"{int(r*100)} " + r"\%",
            #    color=colors[dr_ind],
            # )
        # remove useless lines
        # axs[ind].spines['top'].set_visible(False)
        # axs[ind].spines['right'].set_visible(False)

        # set axis labels&ticks
        axs[ind].set_xlabel(labels["nodes_per_res"])
        axs[ind].set_xscale("log")
        axs[ind].set_xticks([100, 1000, 8000])
        axs[ind].set_xticklabels(["100", "1000", "8000"])

        axs[ind].set_xlim(100, 8000)
        axs[ind].set_ylim(0, 10)
        axs[ind].set_yticks([0, 5, 10])
        # axs[ind].set_yscale("log")

        # axs[ind].xaxis.set_major_formatter(ticker.ScalarFormatter())

    axs[0].set_ylabel(labels["valid_time"])

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
        ncol=5,
        handlelength=0.8,
        handletextpad=0.5,
        columnspacing=1,
    )
    # ax.legend(title="", bbox_to_anchor=(0, 1), ncols=3, loc="lower left")
    set_figure_index(axs, x_direction=0.05, facecol="lightgray")
    # plt.show()
    plt.savefig(
        f"{str(Config.get_git_root())}/Figures/Manuscript/FinalResults_{parres}.pdf",
    )
    print(
        f"Saved plot at {str(Config.get_git_root())}/Figures/Manuscript/FinalResults_{parres}.pdf"
    )


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
    for parres in ["[2]", "[4]", "[8]", "[16]", "[32]"]:
        dr_performance(dfgrouped, parres=parres)


if __name__ == "__main__":
    main()
