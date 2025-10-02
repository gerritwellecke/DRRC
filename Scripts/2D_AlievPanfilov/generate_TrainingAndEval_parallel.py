import multiprocessing as mp
from argparse import ArgumentParser

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


def create_dataset(nr, seed):
    propper_data_needed = True
    vars, t = np.empty(1), np.empty(1)

    while propper_data_needed:
        vars0 = ses.IntegrationMethods.exp_euler_itt(dif, lm, bc, par, seed)
        # Create dataset
        vars, t = ses.IntegrationMethods.exp_euler_itt_with_save(
            dif, lm, bc, par, vars0, prog_feetback=False
        )
        if np.max(vars[-1, 0]) > 0.01:
            propper_data_needed = False
        else:
            seed = np.random.randint(0, 2**32 - 1)
            print(f"\nRedoing simulation nr {nr} with different seed: {seed}\n")
    # Save Dataset
    print(
        "Saving Data at "
        + gitroot
        + "/Data/"
        + f"{conf['Data']['model_dimension']}D_{conf['Data']['model_name']}/"
        + f"{Key}Data{nr}{conf['Data']['Usage']['fileformat']}"
    )
    np.savez_compressed(
        gitroot
        + "/Data/"
        + f"{conf['Data']['model_dimension']}D_{conf['Data']['model_name']}/"
        + f"{Key}Data{nr}{conf['Data']['Usage']['fileformat']}",
        vars=vars,
        t=t,
    )


kernels = 25
for Key in ["Evaluation", "Training"]:
    print(f"Creating {Key} Data.")
    if Key == "Training":
        key = "training"
    else:
        key = "evaluation"
    par["TemporalParameter"]["T"] = par["TemporalParameter"][key + "_length"]
    par["TemporalParameter"]["Transient"] = par["TemporalParameter"][key + "_transient"]
    with mp.Pool(kernels) as pool:
        args = zip(
            np.arange(par["datasets"]),
            np.random.randint(0, 2**32 - 1, size=(par["datasets"],)),
        )
        for res in pool.starmap(create_dataset, args):
            pass
