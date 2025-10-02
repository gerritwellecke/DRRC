import logging as log
import multiprocessing as mp

import numpy as np

from drrc.config import Config as Config

log.basicConfig(level=log.INFO)

# number of datasets
datasets = 50

# get min, max and Data:
min, max = np.inf, -np.inf
Training_min, Training_max = [np.inf for _ in range(datasets)], [
    -np.inf for _ in range(datasets)
]
Evaluation_min, Evaluation_max = [np.inf for _ in range(datasets)], [
    -np.inf for _ in range(datasets)
]

TrainingData = [[] for _ in range(datasets)]
EvaluationData = [[] for _ in range(datasets)]


def load_dataset(nr, Key):
    log.info(f"Loading {Key}Data{nr}.")
    data = np.load(
        f"{str(Config.get_git_root())}/Data/2D_AlievPanfilov/{Key}Data{str(nr)}.npz"
    )
    min = np.min(data["vars"][:, 1])
    max = np.max(data["vars"][:, 1])
    # log.info(f"Min, Max is {min}, {max}")
    return min, max


def load_normalize_save_dataset(nr, Key):
    log.info(f"Loading {Key}Data{nr}.")
    data = np.load(
        f"{str(Config.get_git_root())}/Data/2D_AlievPanfilov/{Key}Data{str(nr)}.npz"
    )
    min = np.min(data["vars"][:, 1])
    max = np.max(data["vars"][:, 1])
    # log.info(f"Min, Max before normalization is {min}, {max}")
    new_vars = np.empty(data["vars"].shape)
    new_vars[:, 0] = data["vars"][:, 0]
    new_vars[:, 1] = (data["vars"][:, 1] - g_min) / (g_max - g_min)
    log.info(
        f"Min, Max after normalization is {np.min(new_vars[:,1])}, {np.max(new_vars[:,1])}"
    )
    log.info(f"Saving {Key}Data{nr}")
    np.savez_compressed(
        f"{str(Config.get_git_root())}/Data/2D_AlievPanfilov_normalized/{Key}Data{str(nr)}.npz",
        vars=new_vars,
        t=data["t"],
    )


# get minimum and maximum of all datasets
mins = []
maxs = []

kernels = 25
for Key in ["Evaluation", "Training"]:
    log.info(f"Loading {Key} Data.")
    with mp.Pool(kernels) as pool:
        if Key == "Training":
            args = zip(
                np.arange(datasets),
                [Key] * datasets,
            )
        else:
            args = zip(
                np.arange(datasets),
                [Key] * datasets,
            )
        for res in pool.starmap(load_dataset, args):
            mins.append(res[0])
            maxs.append(res[1])

global g_min, g_max
g_min = np.min(mins)
g_max = np.max(maxs)

log.info(f"Global Min is {g_min} and Global Max is {g_max}")

# normalizing and saving datasets
for Key in ["Evaluation", "Training"]:
    log.info(f"Loading {Key} Data.")
    with mp.Pool(kernels) as pool:
        if Key == "Training":
            args = zip(
                np.arange(datasets),
                [Key] * datasets,
            )
        else:
            args = zip(
                np.arange(datasets),
                [Key] * datasets,
            )
        for res in pool.starmap(load_normalize_save_dataset, args):
            pass
