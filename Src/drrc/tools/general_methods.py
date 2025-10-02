"""
.. codeauthor:: Luk Fleddermann

This file contains general (usefull) classes and functions,
which can be used in a more general application than the
simulation and prediction of spatially extended systems.
"""

import os
from argparse import ArgumentParser

import h5py
import numpy as np
import yaml
from sympy import ShapeError

# TODO this module does not have one clear purpose -> clean up
# TODO unify yaml, hdf with Config


class progress_feedback(object):
    """
    The class includes different featback functions which can be used to obtain featback
    for itterative aplications of similar steps.
    Some of which are reused from userbased function definitions.
    """

    @staticmethod
    def printProgressBar(
        iteration,
        total,
        prefix="",
        suffix="",
        decimals=1,
        length=100,
        fill="█",
        printEnd="\r",
        name=None,
    ):
        """
        Args:
            iteration: current iteration (Int)
            total: total iterations (Int)
            prefix: prefix string (Str)
            suffix: suffix string (Str)
            decimals: positive number of decimals in percent complete (Int)
            length: character length of bar (Int)
            fill: bar fill character (Str)
            printEnd: end character (e.g. "\r", "\r\n") (Str)

        Notes:
            - The function returns in the last step simultaneously 100% and 100%-10^(-decimals)*1%.
            - The function is a modified version of the function from the source:
              https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console (last visited 30.11.21).
        """
        percent = ("{0:." + str(decimals) + "f}").format(
            100 * (iteration / float(total))
        )
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + "-" * (length - filledLength)
        if name is not None:
            print(f"\r{name}\t{prefix} |{bar}| {percent}% {suffix}", end=printEnd)
        else:
            print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)
        # Print New Line on Complete
        if iteration == total - 1:
            percent = ("{0:." + str(decimals) + "f}").format(100)
            filledLength = length
            bar = fill * filledLength + "-" * (length - filledLength)
            if name is not None:
                print(f"\r{name}\t{prefix} |{bar}| {percent}% {suffix}")
            else:
                print(f"\r{prefix} |{bar}| {percent}% {suffix}")

    @staticmethod
    def print_percent_of_saved_frames(i, n, printEnd="\r"):
        """
        Args:
            iteration: current iteration (Int)
            total: total iterations (Int)
        """
        if (100 * i % n) == 0:
            print(f"Saved {100*i/n} % of the frames.", end=printEnd)


# TODO join this with Config
# - migrate whatever functionality from here is missing in Config to Config
class yml:
    # TODO this is a fairly useless wrapper
    @staticmethod
    def read(PATH):
        """
        Opens a yaml file.

        Args:
            PATH: Path of file to be loaded (str)

        Returns:
            yaml_file if PATH variable is Path to yaml file, else raises Typeerror
        """
        try:
            with open(PATH, "r") as file:
                yaml_file = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(
                PATH + " not found. Provide a path to an existing '.yml' File"
            )
            pass
        return yaml_file

    # TODO Not such a bad idea, but PATH is never checked if it's actually a YAML...
    @staticmethod
    def create(PATH, dict=None):
        """
        Creates a yaml file from a dictionary

        Args:
            PATH: Path (with name) where the yaml file is saved(str)
            dict: Data to store in $PATH/$NAME.yml(dict)
        """
        try:
            with open(PATH, "w") as file:
                yaml.dump(dict, file)
        except FileNotFoundError:
            raise FileNotFoundError(PATH + " not found. Provide an existing Path.")

    # TODO the elif chain in this hurts my head... there has to be a better way
    @staticmethod
    def change(
        PATH, action, variable: list | None = None, replacement=None, NewSavingPath=None
    ):
        """
        changes an existing yaml file.

        Args:
            PATH:
                Path of file to be loaded and edited
            action:
                'c' : change an existing variable
                'a' : add new variable
                'rm': remove existing variable
            variable:
                Ordered Array of names to the variable to work on. (str array)
            replacement:
                For add or change the entrie will be written in dict[variable]
                recommended type: dict or value (Int,float,str)
            NewSavingPath:
                If not None, the changes are saved in a new file with the here saved
                path (+name)


        Returns:
            yaml_file if PATH variable is Path to yaml file, else raises Typeerror

        Notes:
            possibly way faster, files are overwritten with new format
        """
        try:
            with open(PATH, "r+") as file:
                yaml_file = yaml.safe_load(file)

        except FileNotFoundError or PATH[len(PATH) - 4 :] == ".yml":
            raise FileNotFoundError(
                PATH + " not found. Provide a path to an existing '.yml' File"
            )
            pass
        if action == "c":
            if np.shape(variable) == (1,):
                yaml_file[variable[0]] = replacement
            elif np.shape(variable) == (2,):
                yaml_file[variable[0]][variable[1]] = replacement
            elif np.shape(variable) == (3,):
                yaml_file[variable[0]][variable[1]][variable[2]] = replacement
            elif np.shape(variable) == (4,):
                yaml_file[variable[0]][variable[1]][variable[2]][
                    variable[3]
                ] = replacement
            else:
                raise TypeError(
                    "Possible Errors follow:\n"
                    "'variable' and 'replacement' need to be defined for this action.\n"
                    "'variable' must be of type 'str array'\n"
                    "Only for levels of substructures have been added in the code. More can be included in yamlutils.py"
                )
            if NewSavingPath is None:
                yml.create(PATH, yaml_file)
            else:
                yml.create(NewSavingPath, yaml_file)
            return yaml_file

        elif action == "a":
            if np.shape(variable) == (1,):
                if variable[0] in yaml_file:
                    raise ValueError(
                        "The variable exitst already. Use 'c' to change it"
                    )
                else:
                    yaml_file[variable[0]] = replacement
            elif np.shape(variable) == (2,):
                if variable[0] in yaml_file:
                    if variable[1] in yaml_file[variable[0]]:
                        raise ValueError(
                            "The variable exitst already. Use 'c' to change it"
                        )
                    else:
                        yaml_file[variable[0]][variable[1]] = replacement
                else:
                    yaml_file[variable[0]] = {variable[1]: replacement}
            elif np.shape(variable) == (3,):
                if variable[0] in yaml_file:
                    if variable[1] in yaml_file[variable[0]]:
                        if variable[2] in yaml_file[variable[0]][variable[1]]:
                            raise ValueError(
                                "The variable exitst already. Use 'c' to change it"
                            )
                        else:
                            yaml_file[variable[0]][variable[1]][
                                variable[2]
                            ] = replacement
                    else:
                        yaml_file[variable[0]][variable[1]] = {variable[2]: replacement}
                else:
                    yaml_file[variable[0]] = {variable[1]: {variable[2]: replacement}}
            elif np.shape(variable) == (4,):
                if variable[0] in yaml_file:
                    if variable[1] in yaml_file[variable[0]]:
                        if variable[2] in yaml_file[variable[0]][variable[1]]:
                            if (
                                variable[3]
                                in yaml_file[variable[0]][variable[1]][variable[2]]
                            ):
                                raise ValueError(
                                    "The variable exitst already. Use 'c' to change it"
                                )
                            else:
                                yaml_file[variable[0]][variable[1]][variable[2]][
                                    variable[3]
                                ] = replacement
                        else:
                            yaml_file[variable[0]][variable[1]][variable[2]] = {
                                variable[3]: replacement
                            }
                    else:
                        yaml_file[variable[0]][variable[1]] = {
                            variable[2]: {variable[3]: replacement}
                        }
                else:
                    yaml_file[variable[0]] = {
                        variable[1]: {variable[2]: {variable[3]: replacement}}
                    }
            else:
                raise TypeError(
                    "Possible Errors follow:\n"
                    "'variable' and 'replacement' need to be defined for this action.\n"
                    "'variable' must be of type 'str array'\n"
                    "Only for levels of substructures have been added in the code. More can be included in general_methods.py"
                )
            if NewSavingPath is None:
                yml.create(PATH, yaml_file)
            else:
                yml.create(NewSavingPath, yaml_file)

        elif action == "rm":
            if np.shape(variable) == (1,):
                del yaml_file[variable[0]]
            elif np.shape(variable) == (2,):
                del yaml_file[variable[0]][variable[1]]
            elif np.shape(variable) == (3,):
                del yaml_file[variable[0]][variable[1]][variable[2]]
            elif np.shape(variable) == (4,):
                del yaml_file[variable[0]][variable[1]][variable[2]][variable[3]]
            else:
                raise TypeError(
                    "Possible Errors follow:\n"
                    "'variable' and 'replacement' need to be defined for this action.\n"
                    "'variable' must be of type 'str array'\n"
                    "Only for levels of substructures have been added in the code. More can be included in yamlutils.py"
                )
            if NewSavingPath is None:
                yml.create(PATH, yaml_file)
            else:
                yml.create(NewSavingPath, yaml_file)

        else:
            raise TypeError(
                "Variable 'action' is '"
                + action
                + "' but needs to be 'c' to change a variable, 'a' to add a variable or 'rm' to rm a variable."
            )

        return yaml_file


# TODO make this a useful class -> more than just a single method within a class
class hdf5:
    @staticmethod
    def spatiotemp_create(hdf5Path: str, ymlPath: str, redefine: bool = False):
        """
        Creates a hdf5-File for spatiotemporal data at $hdf5Path

        Args:
            hdf5Path: Path where .hdf5 file is saved (str)
            ymlPath: Path to .yml file with all information for the system (str)
            redefine: If hdf5 file already exists, it can be redifined (bool)

        Return:
            hdf5Path: Path tohdf5 file
        """
        if not os.path.exists(ymlPath):
            raise ValueError("yml file at ", ymlPath, " does not exist!")
        if os.path.exists(hdf5Path) and not redefine:
            print(hdf5Path, "exists.")
            print("Writing to the same directory.")
            return hdf5Path
        Par = yml.read(ymlPath)
        Int = Par["TemporalIntegration"]
        hdf5file = h5py.File(hdf5Path, "w")
        hdf5file.create_dataset(
            "TotalFrameNumber", data=(Int["IntegrationSteps"] // Int["SaveEach"])
        )
        hdf5file.create_dataset(
            "SpatialDimension", data=Par["SpatialDiscretization"]["SpatialDimension"]
        )
        hdf5file.create_dataset(
            "Warning",
            data="Dataset is scaled! u_real = (u+16384)/32768 and w_Real = 10*(w+16384)/32768",
        )
        hdf5file.create_dataset(
            "t", data=np.zeros((Int["IntegrationSteps"] // Int["SaveEach"]))
        )

        u_Hdf5 = hdf5file.create_group("u")
        u_time = u_Hdf5.create_group("FastTime")
        u_space = u_Hdf5.create_group("FastSpace")

        w_Hdf5 = hdf5file.create_group("w")
        w_time = w_Hdf5.create_group("FastTime")
        w_space = w_Hdf5.create_group("FastSpace")

        for j in range(hdf5file["TotalFrameNumber"][()]):
            u_space.create_dataset(
                str(j),
                data=np.zeros(
                    (hdf5file["SpatialDimension"][()], hdf5file["SpatialDimension"][()])
                ).astype("int16"),
                compression="gzip",
            )
            w_space.create_dataset(
                str(j),
                data=np.zeros(
                    (hdf5file["SpatialDimension"][()], hdf5file["SpatialDimension"][()])
                ).astype("int16"),
                compression="gzip",
            )

        for j in range(Par["SpatialDiscretization"]["SpatialDimension"] ** 2):
            u_time.create_dataset(
                str(j),
                data=np.zeros((Int["IntegrationSteps"] // Int["SaveEach"])).astype(
                    "int16"
                ),
                compression="gzip",
            )
            w_time.create_dataset(
                str(j),
                data=np.zeros((Int["IntegrationSteps"] // Int["SaveEach"])).astype(
                    "int16"
                ),
                compression="gzip",
            )
        return hdf5Path


def error_calculation(
    Prediction: np.ndarray, Truth: np.ndarray, mean_value: np.ndarray | None = None
):
    """Calculates the Mean-Sqauare-Error, Root-Mean-Square-Error and the Normalised-Root-Mean-Square-Error between two given arrays and a mean_value.

    Args:
        Prediction:
            Array from which the deviation to the second array is measured.
        truth:
            The truth from which the deviation is measured.
        mean_value:
            The temporal mean value of a the truth. Optionally also a mean over
            systemdimensions can be taken reducing the mean_value to a float number.
            If 'None' the mean value of truth (over temp. domain only) is taken.

    Return:
        - Mean-Sqauare-Error
        - Root-Mean-Square-Error
        - Normalised-Root-Mean-Square-Error
    """
    if np.shape(Prediction) != np.shape(Truth):
        raise ShapeError(
            " and Prediction and Truth need to have the same shape, but Prediction.shape is {} and Truth.shape is {}".format(
                Prediction.shape, Truth.shape
            )
        )
    else:
        if mean_value is None:
            mean_value = np.mean(Truth, axis=0)
        ########If temporal resolution of the error is required:###########
        # DimNr = np.shape(np.shape(Truth))[0]
        # ax = tuple(np.arange(1, DimNr))
        # MSE_t = (1/np.prod(np.shape(Truth)[1:]))*np.sum((Truth-Prediction)**2, axis=ax)
        MSE = 1 / np.size(Truth) * np.sum((Truth - Prediction) ** 2)
        RMSE = np.sqrt(MSE)
        MSE_mean = 1 / np.size(Truth) * np.sum((Truth - mean_value) ** 2)
        if MSE_mean == 0:
            NRMSE = np.inf
        else:
            NRMSE = np.sqrt(MSE / MSE_mean)
        return (MSE, RMSE, NRMSE)


def _get_path():
    """
    Returns:
        current Path
    """
    return os.path.dirname(os.path.abspath(__file__)) + "/"
