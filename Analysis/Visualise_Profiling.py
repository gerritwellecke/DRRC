import logging
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from drrc.analysis import AutomaticPostprocessing
from drrc.config import Config


def concatenate_csv_files(glob_pattern) -> pd.DataFrame:
    """
    Concatenates multiple CSV files into a single DataFrame.

    Args:
        glob_pattern (str): The glob pattern used to match CSV files.

    Returns:
        pandas.DataFrame: A DataFrame resulting from the concatenation of all matched
            CSV files, or None if no files match the glob pattern.

    Raises:
        ValueError: If the glob pattern does not match any files.
    """
    # Get a list of all the csv files matching the glob pattern
    csv_files = glob(glob_pattern, recursive=True)
    # Concatenate all the dataframes into one
    concatenated_df = pd.concat(
        (pd.read_csv(fp) for fp in tqdm(csv_files, desc=f"Creating DataFrame")),
        ignore_index=True,
    )
    return concatenated_df


def visualize_runtime(system="1D_KuramotoSivashinsky"):
    markers = ["1", "2", "3", "4", "+", "x"]
    df = concatenate_csv_files(
        str(Config.get_git_root()) + f"/Data/RunTimes/{system}/**/DataFrame.csv"
    )

    parallel_reservoir_grid_shapes = df["parallelreservoirs_grid_shape"].unique()
    if system == "1D_KuramotoSivashinsky":
        trainingsdomain = "[0]"
    elif system == "2D_AlievPanfilov":
        trainingsdomain = "[0, 0]"
    else:
        raise ValueError("System not recognized")

    #### Plot runtime maxima comparing different trainings
    for transformation in ["identity", "pca", "fft"]:
        df_grouped = df[
            (df["Transformation"] == transformation)
            & (df["identical_adjacencymatrix"] == True)
            & (df["identical_inputmatrix"] == True)
        ]
        logging.info(
            f"Number of rows for transformation {transformation} is: {len(df_grouped)}"
        )

        for runtime_specifier in [
            "total_RunTime",
            "PredictionTime",
            "TransientUpdateTime",
            "ReservoirTrainTime",
        ]:
            df_grouped_train = (
                df_grouped.groupby(["reservoir_nodes", "identical_outputmatrix"])[
                    runtime_specifier
                ]
                .max()
                .reset_index()
            )
            fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
            ax.set_ylabel(f"{runtime_specifier} [s]")
            for train in [trainingsdomain, "False", "combine_data"]:
                ax.plot(
                    df_grouped_train[
                        (df_grouped_train["identical_outputmatrix"] == train)
                    ]["reservoir_nodes"],
                    df_grouped_train[
                        (df_grouped_train["identical_outputmatrix"] == train)
                    ][runtime_specifier],
                    label=f"{train}",
                )
            ax.set_xlabel("Reservoir Nodes")
            ax.set_title(f"{transformation}")
            ax.legend(title="Individual Training", frameon=False)
            plt.tight_layout()
            plt.savefig(
                f"{str(Config.get_git_root())}/Figures/RunTimes/{system}/{transformation}_CompareTraining_{runtime_specifier}.pdf"
            )
            plt.close()
            logging.info(
                f"Saved figure at {str(Config.get_git_root())}/Figures/RunTimes/{system}/{transformation}_CompareTraining_{runtime_specifier}.pdf"
            )

    #### Plot runtime maxima using only one training
    for transformation in ["identity", "pca", "fft"]:
        df_grouped = df[
            (df["Transformation"] == transformation)
            & (df["identical_outputmatrix"] == trainingsdomain)
            & (df["identical_adjacencymatrix"] == True)
            & (df["identical_inputmatrix"] == True)
        ]
        logging.info(
            f"Number of rows for transformation {transformation} with train {trainingsdomain}: {len(df_grouped)}"
        )

        for runtime_specifier in [
            "total_RunTime",
            "PredictionTime",
            "TransientUpdateTime",
            "ReservoirTrainTime",
        ]:
            df_grouped_adjacency = (
                df_grouped.groupby(["reservoir_nodes", "adjacency_degree"])[
                    runtime_specifier
                ]
                .max()
                .reset_index()
            )
            df_grouped_parres = (
                df_grouped.groupby(
                    ["reservoir_nodes", "parallelreservoirs_grid_shape"]
                )[runtime_specifier]
                .max()
                .reset_index()
            )
            df_grouped_ghosts = (
                df_grouped.groupby(["reservoir_nodes", "parallelreservoirs_ghosts"])[
                    runtime_specifier
                ]
                .max()
                .reset_index()
            )

            if transformation == "identity":
                fig, ax = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
            else:
                fig, ax = plt.subplots(1, 4, figsize=(13.5, 3.5), sharey=True)
                df_grouped_drfraction = (
                    df_grouped.groupby(
                        ["reservoir_nodes", "dimensionreduction_fraction"]
                    )[runtime_specifier]
                    .max()
                    .reset_index()
                )
                ax[3].set_title("Dimension Reduction Fraction")
                for n, frac in enumerate([0.25, 0.5, 0.75, 1.0]):
                    ax[3].scatter(
                        df_grouped_drfraction[
                            (
                                df_grouped_drfraction["dimensionreduction_fraction"]
                                == frac
                            )
                        ]["reservoir_nodes"],
                        df_grouped_drfraction[
                            (
                                df_grouped_drfraction["dimensionreduction_fraction"]
                                == frac
                            )
                        ][runtime_specifier],
                        marker=markers[n],
                        label=f"DR Fraction {frac}",
                    )
                ax[3].legend(frameon=False)

            ax[0].set_ylabel(f"{runtime_specifier} [s]")

            ax[0].set_title("Grid Shape")
            for n, grid_shape in enumerate(parallel_reservoir_grid_shapes):
                ax[0].scatter(
                    df_grouped_parres[
                        (
                            df_grouped_parres["parallelreservoirs_grid_shape"]
                            == grid_shape
                        )
                    ]["reservoir_nodes"],
                    df_grouped_parres[
                        (
                            df_grouped_parres["parallelreservoirs_grid_shape"]
                            == grid_shape
                        )
                    ][runtime_specifier],
                    marker=markers[n],
                    label=f"Grid Shape {grid_shape}",
                )
            ax[0].legend(frameon=False)

            ax[1].set_title("Adjacency Degree")
            for n, adj_degree in enumerate([2, 4]):
                ax[1].scatter(
                    df_grouped_adjacency[
                        (df_grouped_adjacency["adjacency_degree"] == adj_degree)
                    ]["reservoir_nodes"],
                    df_grouped_adjacency[
                        (df_grouped_adjacency["adjacency_degree"] == adj_degree)
                    ][runtime_specifier],
                    marker=markers[n],
                    label=f"Adjacency Degree {adj_degree}",
                )
            ax[1].legend(frameon=False)

            ax[2].set_title("Parallel Reservoir Ghosts")
            for n, ghosts in enumerate([2, 6, 10]):
                ax[2].scatter(
                    df_grouped_ghosts[
                        (df_grouped_ghosts["parallelreservoirs_ghosts"] == ghosts)
                    ]["reservoir_nodes"],
                    df_grouped_ghosts[
                        (df_grouped_ghosts["parallelreservoirs_ghosts"] == ghosts)
                    ][runtime_specifier],
                    marker=markers[n],
                    label=f"Ghost Cells {ghosts}",
                )
            ax[2].legend(frameon=False)

            for a in ax:
                a.set_xlabel("Reservoir Nodes")
            plt.tight_layout()
            plt.savefig(
                f"{str(Config.get_git_root())}/Figures/RunTimes/{system}/{transformation}_TrainOn0_{runtime_specifier}.pdf"
            )
            plt.close()
            logging.info(
                f"Saved figure at {str(Config.get_git_root())}/Figures/RunTimes/{system}/{transformation}_TrainOn0_{runtime_specifier}.pdf"
            )


def visualize_memory(system="1D_KuramotoSivashinsky"):
    markers = ["1", "2", "3", "4", "+", "x"]
    df = concatenate_csv_files(
        str(Config.get_git_root()) + f"/Data/Memory/{system}/**/DataFrame.csv"
    )  # .dropna()
    parallel_reservoir_grid_shapes = df["parallelreservoirs_grid_shape"].unique()
    if system == "1D_KuramotoSivashinsky":
        trainingsdomain = "[0]"
    elif system == "2D_AlievPanfilov":
        trainingsdomain = "[0, 0]"
    else:
        raise ValueError("System not recognized")

    # print(df[df["reservoir_nodes"]==4000].groupby(["transformation", "identical_outputmatrix", "parallelreservoirs_grid_shape"])["Memory_total_max"].max())
    # print(df["identical_adjacencymatrix"].unique())
    # print(df["identical_inputmatrix"].unique())
    # print(df["transformation"].unique())
    # print(len(df))
    #### Plot memory maxima
    for transformation in ["identity", "pca", "fft"]:
        df_grouped = df[
            (df["Transformation"] == transformation)
            & (df["identical_outputmatrix"] == trainingsdomain)
            & (df["identical_adjacencymatrix"] == True)
            & (df["identical_inputmatrix"] == True)
        ]
        # print(transformation, len(df_grouped))
        for memory_specifier in [
            "Memory_train",
            "Memory_transient",
            "Memory_predict",
            "Memory_total_max",
        ]:
            df_grouped_adjacency = (
                df_grouped.groupby(["reservoir_nodes", "adjacency_degree"])[
                    memory_specifier
                ]
                .max()
                .reset_index()
            )
            df_grouped_parres = (
                df_grouped.groupby(
                    ["reservoir_nodes", "parallelreservoirs_grid_shape"]
                )[memory_specifier]
                .max()
                .reset_index()
            )
            df_grouped_ghosts = (
                df_grouped.groupby(["reservoir_nodes", "parallelreservoirs_ghosts"])[
                    memory_specifier
                ]
                .max()
                .reset_index()
            )

            if transformation == "identity":
                fig, ax = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
            else:
                fig, ax = plt.subplots(1, 4, figsize=(13.5, 3.5), sharey=True)
                df_grouped_drfraction = (
                    df_grouped.groupby(
                        ["reservoir_nodes", "dimensionreduction_fraction"]
                    )[memory_specifier]
                    .max()
                    .reset_index()
                )
                ax[3].set_title("Dimension Reduction Fraction")
                for n, frac in enumerate([0.25, 0.5, 0.75, 1.0]):
                    ax[3].scatter(
                        df_grouped_drfraction[
                            (
                                df_grouped_drfraction["dimensionreduction_fraction"]
                                == frac
                            )
                        ]["reservoir_nodes"],
                        df_grouped_drfraction[
                            (
                                df_grouped_drfraction["dimensionreduction_fraction"]
                                == frac
                            )
                        ][memory_specifier]
                        / 1000,
                        marker=markers[n],
                        label=f"DR Fraction {frac}",
                    )
                ax[3].legend(frameon=False)

            ax[0].set_ylabel(f"{memory_specifier} [GB]")

            ax[0].set_title("Grid Shape")
            for n, grid_shape in enumerate(parallel_reservoir_grid_shapes):
                ax[0].scatter(
                    df_grouped_parres[
                        (
                            df_grouped_parres["parallelreservoirs_grid_shape"]
                            == grid_shape
                        )
                    ]["reservoir_nodes"],
                    df_grouped_parres[
                        (
                            df_grouped_parres["parallelreservoirs_grid_shape"]
                            == grid_shape
                        )
                    ][memory_specifier]
                    / 1000,
                    marker=markers[n],
                    label=f"Grid Shape {grid_shape}",
                )
            ax[0].legend(frameon=False)

            ax[1].set_title("Adjacency Degree")
            for n, adj_degree in enumerate([2, 4]):
                ax[1].scatter(
                    df_grouped_adjacency[
                        (df_grouped_adjacency["adjacency_degree"] == adj_degree)
                    ]["reservoir_nodes"],
                    df_grouped_adjacency[
                        (df_grouped_adjacency["adjacency_degree"] == adj_degree)
                    ][memory_specifier]
                    / 1000,
                    marker=markers[n],
                    label=f"Adjacency Degree {adj_degree}",
                )
            ax[1].legend(frameon=False)

            ax[2].set_title("Parallel Reservoir Ghosts")
            for n, ghosts in enumerate([2, 6, 10]):
                ax[2].scatter(
                    df_grouped_ghosts[
                        (df_grouped_ghosts["parallelreservoirs_ghosts"] == ghosts)
                    ]["reservoir_nodes"],
                    df_grouped_ghosts[
                        (df_grouped_ghosts["parallelreservoirs_ghosts"] == ghosts)
                    ][memory_specifier]
                    / 1000,
                    marker=markers[n],
                    label=f"Ghost Cells {ghosts}",
                )
            ax[2].legend(frameon=False)
            for a in ax:
                a.set_xlabel("Reservoir Nodes")
            plt.tight_layout()
            plt.savefig(
                f"{str(Config.get_git_root())}/Figures/Memory/{system}/{transformation}_TrainOn0_{memory_specifier}.pdf"
            )
            plt.close()
            logging.info(
                f"Saved figure at {str(Config.get_git_root())}/Figures/Memory/{system}/{transformation}_TrainOn0_{memory_specifier}.pdf"
            )

    # fig, ax = plt.subplots(1, 4, figsize=(10, 3.5), sharey=True)
    # ax[0].set_ylabel('Maximal Memory [MB]')
    # for n, grid_shape in enumerate([(1, 1), (2, 2), (4, 4), (8, 8)]):
    df = df[(df["identical_outputmatrix"] == trainingsdomain)]
    dfgrouped = df.loc[
        df.groupby(
            [
                "reservoir_nodes",
                "Transformation",
                "dimensionreduction_fraction",
                # "parallelreservoirs_grid_shape",
            ]
        )["Memory_total_max"].idxmax()
    ]

    df_identity = dfgrouped[(dfgrouped["Transformation"] == "identity")]
    df_pca = dfgrouped[(dfgrouped["Transformation"] == "pca")]
    df_fft = dfgrouped[(dfgrouped["Transformation"] == "fft")]

    # logging.info(df_identity)

    fig, ax = plt.subplots(1, 4, figsize=(10, 3.5), sharey=True)
    ax[0].set_ylabel("Maximal Memory [GB]")
    for n, frac in enumerate([1.0, 0.75, 0.5, 0.25]):
        ax[n].set_title(f"{int(100*frac)}% ")
        if frac == 1.0:
            ax[n].scatter(
                df_identity[(df_identity["dimensionreduction_fraction"] == frac)][
                    "reservoir_nodes"
                ],
                df_identity[(df_identity["dimensionreduction_fraction"] == frac)][
                    "Memory_total_max"
                ]
                / 1000,
                label="identity",
                c="tab:blue",
                marker="1",
            )
        # print(df_fft[(df_fft["dimensionreduction_fraction"] == frac)]["reservoir_nodes"])
        ax[n].scatter(
            df_pca[(df_pca["dimensionreduction_fraction"] == frac)]["reservoir_nodes"],
            df_pca[(df_pca["dimensionreduction_fraction"] == frac)]["Memory_total_max"]
            / 1000,
            label="pca",
            c="tab:orange",
            marker="2",
        )
        ax[n].scatter(
            df_fft[(df_fft["dimensionreduction_fraction"] == frac)]["reservoir_nodes"],
            df_fft[(df_fft["dimensionreduction_fraction"] == frac)]["Memory_total_max"]
            / 1000,
            label="fft",
            c="tab:green",
            marker="3",
        )
    ax[0].legend(frameon=False)
    #
    #    ax[n].legend(title="transformation", frameon=False)
    #
    plt.tight_layout()
    plt.savefig(
        f"{str(Config.get_git_root())}/Figures/Memory/{system}/Memory_total_max.pdf"
    )
    plt.close()
    logging.info(
        f"saved figure at {str(Config.get_git_root())}/Figures/Memory/{system}/Memory_total_max.pdf"
    )


def main():
    logging.basicConfig(level=logging.INFO)

    for Key in ["Memory", "RunTimes"]:
        AutomaticPostprocessing(
            Config.get_git_root() / Path(f"Data/{Key}")
        ).auto_concatenate()

    for system in ["1D_KuramotoSivashinsky", "2D_AlievPanfilov"]:
        try:
            visualize_memory(system)
        except ValueError:
            logging.info(f"Could not visualize memory for {system}")
        try:
            visualize_runtime(system)
        except ValueError:
            logging.info(f"Could not visualize runtime for {system}")


if __name__ == "__main__":
    main()
