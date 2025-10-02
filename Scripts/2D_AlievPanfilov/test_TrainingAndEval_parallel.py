import logging
import multiprocessing as mp
from argparse import ArgumentParser

import numpy as np

import drrc.spatially_extended_systems as ses

# import drrc.tools.general_methods as gm
from drrc.config import Config
from drrc.tools.visualization import visualization


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


def visualize_dataset(nr):
    print("Loading Data.")
    vars = np.load(
        gitroot
        + "/Data/"
        + f"{conf['Data']['model_dimension']}D_{conf['Data']['model_name']}/"
        + f"{Key}Data{nr}{conf['Data']['Usage']['fileformat']}"
    )["vars"]
    t = np.load(
        gitroot
        + "/Data/"
        + f"{conf['Data']['model_dimension']}D_{conf['Data']['model_name']}/"
        + f"{Key}Data{nr}{conf['Data']['Usage']['fileformat']}"
    )["t"]

    # Plotting
    visualization.produce_animation(
        t,
        par["SystemParameters"]["SpatialParameter"]["physical_length"],
        par["SystemParameters"]["SpatialParameter"]["dx"],
        fps=par["AnimationSaving"]["fps"],
        T_out=par["AnimationSaving"]["T_out"],
        animation_data=[vars[:, 0]],
        bounds=None,
        names=[Key + str(nr)],
        Path=gitroot + "/Figures/2D_AlievPanfilov/TrainingData/",
        prog_feedback=False,
    )
    # testing if it dies
    if np.max(vars[-1, 0]) < 0.1:
        raise TypeError(f"Dynamics in {Key}dataset{nr} dies.")


for Key in ["Training", "Evaluation"]:
    if Key == "Training":
        key = "training"
    else:
        key = "evaluation"
    print(f"Plotting {Key} Data.")
    par["TemporalParameter"]["T"] = par["TemporalParameter"][key + "_length"]
    par["TemporalParameter"]["Transient"] = par["TemporalParameter"][key + "_transient"]
    with mp.Pool(2) as pool:
        args = zip(np.arange(par["datasets"]))
        for res in pool.starmap(visualize_dataset, args):
            pass
