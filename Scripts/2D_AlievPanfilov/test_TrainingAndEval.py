import logging
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

import drrc.spatially_extended_systems as ses
from drrc.config import Config


def get_arguments():
    """Return script arguments as dictionary
    Args:
        - None
    Returns:
        - dictionary containing script arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "PATH", help=f"Absoute Path to parameter file in system or from git root."
    )

    # Potentially more arguments

    return parser.parse_args()


args = get_arguments()
conf = Config(args.PATH)
par = conf["Data"]["Creation"]
gitroot = str(conf.get_git_root()) + "/"

lm = ses.LocalModel(par["SystemParameters"])
dif = ses.Diffusion(par["SystemParameters"])
bc = ses.BoundaryCondition(par["SystemParameters"])
logging.basicConfig(level=logging.INFO)


def visualize_fistandlast_dataset(
    nr: int,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    logging.info("Loading Data.")
    vars = np.load(
        gitroot
        + "/Data/"
        + f"{conf['Data']['model_dimension']}D_{conf['Data']['model_name']}/"
        + f"{Key}Data{nr}{conf['Data']['Usage']['fileformat']}"
    )["vars"]
    if not isinstance(vars, np.ndarray):
        raise ValueError("Data is not a numpy array.")
    t = np.load(
        gitroot
        + "/Data/"
        + f"{conf['Data']['model_dimension']}D_{conf['Data']['model_name']}/"
        + f"{Key}Data{nr}{conf['Data']['Usage']['fileformat']}"
    )["t"]
    # testing if it dies
    if np.max(vars[-1, 0]) < 0.1:
        logging.warning(f"Dynamics in {Key}dataset{nr} dies.")
        return None, None
    else:
        return vars[1, 0], vars[-1, 0]


for Key in ["Evaluation", "Training"]:
    if Key == "Training":
        key = "training"
    else:
        key = "evaluation"
    logging.info(f"Testing {Key} Data.")
    par["TemporalParameter"]["T"] = par["TemporalParameter"][key + "_length"]
    par["TemporalParameter"]["Transient"] = par["TemporalParameter"][key + "_transient"]
    # args = zip(np.arange(par["datasets"]))
    v0 = np.zeros(
        (
            par["datasets"],
            par["SystemParameters"]["SpatialParameter"]["physical_length"],
            par["SystemParameters"]["SpatialParameter"]["physical_length"],
        )
    )
    vN = np.copy(v0)

    if isinstance(par["datasets"], int):
        datasets = par["datasets"]
    else:
        datasets = 1

    for res in np.arange(datasets):
        v0[res], vN[res] = visualize_fistandlast_dataset(res)

    fig, ax = plt.subplots(
        nrows=par["datasets"] // 10,
        ncols=10,
        sharex=True,
        sharey=True,
        figsize=(10, 5),
    )
    vmin, vmax = 0, 1
    for res in np.arange(par["datasets"]):
        ax[res // 10, res % 10].set_title(Key + " " + str(res), pad=0)
        ax[res // 10, res % 10].title.set_size(5)
        ax[res // 10, res % 10].set_xticks([])
        ax[res // 10, res % 10].set_yticks([])

        ax[res // 10, res % 10].imshow(v0[res], vmin=vmin, vmax=vmax)
    # plt.tight_layout()
    plt.subplots_adjust(wspace=-0.2, hspace=0.15)
    plt.savefig(
        gitroot
        + "/Figures/2D_AlievPanfilov/TrainingData/Compare_"
        + Key
        + "_initial.pdf",
        bbox_inches="tight",
    )
    st = (
        "Saved figure at "
        + gitroot
        + "/Figures/2D_AlievPanfilov/TrainingData/Compare_"
        + Key
        + "_initial.pdf"
    )
    logging.info(st)

    fig, ax = plt.subplots(
        nrows=par["datasets"] // 10,
        ncols=10,
        sharex=True,
        sharey=True,
        figsize=(10, 5),
    )
    vmin, vmax = 0, 1
    for res in np.arange(par["datasets"]):
        ax[res // 10, res % 10].set_title(Key + " " + str(res), pad=0)
        ax[res // 10, res % 10].title.set_size(5)
        ax[res // 10, res % 10].set_xticks([])
        ax[res // 10, res % 10].set_yticks([])

        ax[res // 10, res % 10].imshow(vN[res], vmin=vmin, vmax=vmax)
    # plt.tight_layout()
    plt.subplots_adjust(wspace=-0.2, hspace=0.15)
    plt.savefig(
        gitroot
        + "/Figures/2D_AlievPanfilov/TrainingData/Compare_"
        + Key
        + "_final.pdf",
        bbox_inches="tight",
    )
    st = (
        "Saved figure at "
        + gitroot
        + "/Figures/2D_AlievPanfilov/TrainingData/Compare_"
        + Key
        + "_final.pdf"
    )
    logging.info(st)
