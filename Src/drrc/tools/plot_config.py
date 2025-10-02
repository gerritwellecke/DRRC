import matplotlib.pyplot as plt

import drrc


def set_plot_style() -> None:
    """Set matplotlib style for publication-style plots"""
    plt.style.use("drrc.tools.publication")


def set_figure_index(
    axs, x_direction=0.01, y_direction=0.95, *, facecol: str = "white"
) -> None:
    """Set figure index for publication-style plots"""
    labels = ["a", "b", "c", "d"]
    for ind, ax in enumerate(axs):
        ax.text(
            x_direction,
            y_direction,
            r"\textbf{" + str(labels[ind]) + r"}",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            bbox=dict(facecolor=facecol, edgecolor="none", pad=0.2),
        )


# useful global variables ##############################################################
figsize_single = (3.37, 2.08)
figsize_single_tworows = (3.37, 3.37)
figsize_double = (6.69, 2.08)
figsize_huge = (6.69, 4.16)

cmap = "coolwarm"

labels = {
    "nodes_per_res": r"nodes per reservoir $N$",
    "valid_time": r"valid time $t_\mathrm{val} [1/\lambda_\mathrm{max}]$",
    "time": r"time $t[1/\lambda_\mathrm{max}]$",
    "num_par_res": r"number of parallel reservoirs $\#_\mathrm{res}$",
    "space": r"space $x$",
    "relative_performance": r"relative performance $t_\mathrm{val}/t_\mathrm{val}^{\mathrm{id}}$",
}

step_size = 0.25
lyapunov_time = 0.095
