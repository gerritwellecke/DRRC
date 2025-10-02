from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fftfreq

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
    parser.add_argument("PATH", help="Path(+Name) from 'DRRC/' to parameter file.")

    # Potentially more arguments
    return parser.parse_args()


args = get_arguments()
conf = Config(args.PATH)
par = conf["Data"]["Creation"]

# Load Data
print("Loading Data.")
vars = np.load(conf.get_data_dir() / Path("TrainingData0.npz"))["vars"]
t = np.load(conf.get_data_dir() / Path("TrainingData0.npz"))["t"]

# Calculate Fourier Trafo
print("Transforming Data.")
vars_ft = np.fft.ifftshift(vars)
vars_ft = np.fft.fft2(vars_ft, axes=(2, 3))
vars_ft = np.fft.fftshift(vars_ft)

# TransformDomain
xf = fftfreq(
    int(
        par["SystemParameters"]["SpatialParameter"]["physical_length"]
        / par["SystemParameters"]["SpatialParameter"]["dx"]
    ),
    par["SystemParameters"]["SpatialParameter"]["dx"],
)

# ProduceAnimation (This is not done, the animation function is not compatible with fouriertransform...)
visualization.produce_animation(
    t,
    x=xf.max() - xf.min() + xf[1] - xf[0],
    dx=xf[1] - xf[0],
    fps=par["AnimationSaving"]["fps"],
    T_out=par["AnimationSaving"]["T_out"],
    animation_data=[np.abs(vars_ft[:, 0])],
    bounds=None,
    names=[str(par["SystemParameters"]["initial_condition"]) + "_FFT"],
    Path=str(Config.get_git_root()) + "/Figures/2D_AlievPanfilov/",
    prog_feedback=True,
)
print("x and y axes will be off in the animation!")

# ProduceFigures
plt.imshow(
    (np.max(np.log(np.abs(vars_ft[1:, 0])), axis=0)),
    vmin=0,
    vmax=np.log(np.max(np.abs(vars_ft[1:, 0]))),
    interpolation="None",
    extent=(xf.min(), xf.max(), xf.min(), xf.max()),
)
plt.xlabel(r"$k_x$")
plt.ylabel(r"$k_y$")
plt.colorbar(label=r"$\log(\|\hat{F}(u)\|)$")
plt.savefig(
    str(Config.get_git_root())
    + "/Figures/2D_AlievPanfilov/"
    + par["SystemParameters"]["initial_condition"]
    + "_FFT_max.pdf"
)
