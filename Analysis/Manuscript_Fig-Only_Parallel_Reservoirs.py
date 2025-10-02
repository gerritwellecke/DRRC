import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drrc.tools.plot_config import *


def figure_only_parallel_reservoirs(Only5g=False):
    """Generate plot showing parallel reservoir performance without dimensionality reduction"""
    # collect all simulations
    file_list = Path("../Data/ValidTimes/1D_KuramotoSivashinsky/").glob(
        "**/ProcessedValidTimes.csv"
    )
    if Only5g:
        ghosts = 5
        name_append = "_5g"
    else:
        ghosts = 10
        name_append = ""

    # import identity data from all relevant simulations
    df = pd.concat([pd.read_csv(fp) for fp in file_list], ignore_index=True)
    df = df[
        (df["Transformation"] == "identity")
        & (df["adjacency_degree"] == 2)
        & (df["input_bias"] == 0.01)
        & (df["parallelreservoirs_ghosts"] == ghosts)
        & (df["training_output_bias"] == 0.01)
    ]

    # clean up data for plot
    pt = df.pivot_table(
        values="mean_ValidTime",
        index="reservoir_nodes",
        columns="parallelreservoirs_grid_shape",
        aggfunc="max",
    )
    pt = pt.reindex(sorted(pt.columns, key=ast.literal_eval), axis="columns")
    pt.rename(lambda name: str(ast.literal_eval(name)[0]), axis=1, inplace=True)
    # scale with Lyapunov time
    pt = pt * lyapunov_time

    # get colours
    nr_plots = pt.columns.nunique()
    colormap = "viridis"
    colorrange = [0, 0.85]  # control where cmap begins (lighter darker) [0,1]
    colors = plt.get_cmap(colormap)(np.linspace(colorrange[0], colorrange[1], nr_plots))

    # plot the valid time as function of nodes per reservoir
    # one line per number of parallel reservoirs
    ax = plt.figure(figsize=(3.4, 2.2), constrained_layout=True).gca()
    pt.plot(
        ax=ax,
        xlabel=labels["nodes_per_res"],
        ylabel=labels["valid_time"],
        color=colors,
    )
    ax.set_xscale("log")
    xticks = [100, 1000, 8000]
    yticks = [0, 5, 10]
    ax.set_xticks(xticks, labels=xticks)
    ax.set_yticks(yticks, labels=yticks)
    ax.set_xlim(100, 8000)
    ax.legend(
        title="", bbox_to_anchor=(0, 1, 1, 0.1), ncols=nr_plots, loc="upper center"
    )
    plt.savefig(f"../Figures/Manuscript/Only-Parallel-Reservoirs{name_append}.pdf")


if __name__ == "__main__":
    set_plot_style()

    # generate plot
    figure_only_parallel_reservoirs(Only5g=False)
    figure_only_parallel_reservoirs(Only5g=True)
