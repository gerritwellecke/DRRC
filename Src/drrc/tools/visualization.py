"""
.. codeauthor:: Luk Fleddermann
"""
import logging
from inspect import currentframe, getframeinfo

import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.lib.shape_base import dsplit

from drrc.config import Config

from .general_methods import progress_feedback


class visualization(object):
    """
    Visualtizations of temporally extended two-dimensional data.
    """

    @staticmethod
    def produce_animation(
        t: np.ndarray,
        x: float,
        dx: float,
        fps: int,
        T_out: int,
        animation_data: list[np.ndarray],
        bounds: list[list[int]] | None = None,
        names: list[str] = ["test_animation"],
        plt_shape: tuple[int, int] = (1, 1),
        Path: str = f"{Config.get_git_root()}/Figures/",
        figure_name: str | None = None,
        prog_feedback: bool = True,
        one_cbar: bool = False,
    ):
        """
        Produces an animation of arbitrary length and resolution from the given data.

        Args:
            t (np.ndarray): Time array.
            x (float): Physical total length.
            dx (float): Infinitesimal length.
            fps (int): Frames per second of the animation.
            T_out (int): Length of the animation.
            animation_data (list[np.ndarray]): List of Data of excitation over time. The first dimension is temporal. Different list elements are shown in different figures.
            bounds (list[list[int]] | None, optional): Bounds of the range in the heatmap. Defaults to None. Different list elements are used in different figures.
            names (list[str], optional): Names of the animation. Defaults to ["test_animation"]. Different list elements are used in different figures.
            plt_shape (tuple[int, int], optional): Shape of the plot: (nrows, ncols). Defaults to (1, 1).
            Path (str, optional): Path to the directory where the file is saved. Defaults to f"{Config.get_git_root()}/Figures/".
            figure_name (str | None, optional): Name of the figure. Defaults to None.
            prog_feedback (bool, optional): Whether to show progress feedback. Defaults to True.
            one_cbar (bool, optional): Whether to show only one colorbar. Defaults to False.

        Returns:
            mp4 data

        Notes:
            The function has limited application so far and shall be extended to optionally produce multiple plots next to each other.

        Raises:
            TypeError: If animation_data and animation_data2 have different shapes.

        """
        if bounds is None:
            bounds = [[np.min(data), np.max(data)] for data in animation_data]
        elif (np.array(bounds).shape != (len(animation_data), 2)) and (
            np.array(bounds).shape != (1, 2)
        ):
            logging.info(
                f"Warning: Shape of 'bounds' needs to be {(len(animation_data),2)} but is {np.array(bounds).shape} minimum and maximum of data are chosen as minimum."
            )
            bounds = [[np.min(data), np.max(data)] for data in animation_data]

        # errors
        if len(animation_data) > 1:
            for i in range(1, len(animation_data)):
                if animation_data[i].shape != animation_data[0].shape:
                    raise TypeError(
                        f"All sets of animationdata need to be of same shape but animationdata[{i}] is of shape {animation_data[i].shape} and animationdata[0] is of shape {animation_data[0].shape}"
                    )

        if animation_data[0].shape[0] < fps * T_out:
            frame = currentframe()
            if frame is not None:
                frameinfo = getframeinfo(frame)
                raise ValueError(
                    f"In file {frameinfo.filename} line {frameinfo.lineno}. The data has temporally a lower resolution than it is asked for.\n\tIn total ({T_out * fps}) frames shall be portrait, but only {animation_data[0].shape[0]} are given."
                )

        if np.prod(plt_shape) != len(animation_data):
            frame = currentframe()
            if frame is not None:
                frameinfo = getframeinfo(frame)
                raise ValueError(
                    f"In file {frameinfo.filename} line {frameinfo.lineno}. The number of plots in plt_shape does not match the number of given data sets."
                )

        # warnings
        if round(x / dx) != animation_data[0].shape[1]:
            frame = currentframe()
            if frame is not None:
                frameinfo = getframeinfo(frame)
                logging.warning(
                    f"In file {frameinfo.filename} line {frameinfo.lineno}: Shape of animation data, does not match the given physical length 'x' combined with resolution 'dx'.\n\tGiven physical length and resolution imply {round(x / dx)} spacial coordinates but {animation_data[0].shape[1]} coordinates are given in the animation data."
                )

        logging.info(
            f"Producing and saving {len(animation_data)} in shape {plt_shape}."
        )

        fig, axs = plt.subplots(
            nrows=plt_shape[0], ncols=plt_shape[1], sharey=True, sharex=True
        )

        if not one_cbar:
            fig.subplots_adjust(hspace=0.5, wspace=0.5)
        else:
            fig.subplots_adjust(hspace=0.5)

        if np.prod(np.array(plt_shape)) == 1:
            axs = np.array([axs])

        axs = axs.flatten()

        ims = []
        for i in range(len(animation_data)):
            if len(bounds) == 1:
                bound = [bounds[0][0], bounds[0][1]]
            else:
                bound = [bounds[i][0], bounds[i][1]]
            ims.append(
                axs[i].imshow(
                    animation_data[i][0],
                    animated=True,
                    extent=[-x / 2, x / 2, -x / 2, x / 2],
                    interpolation="None",
                    vmin=bound[0],
                    vmax=bound[1],
                )
            )
            if len(names) != 1:
                axs[i].set_title(names[i])
            else:
                axs[i].set_title(names[0])
            if not one_cbar:
                fig.colorbar(
                    ims[i], ax=axs[i], orientation="vertical", location="right"
                )

            if i >= np.prod(plt_shape) - plt_shape[1]:
                axs[i].set_xlabel("x")
            if i % plt_shape[1] == 0:
                axs[i].set_ylabel("y")

        fig.suptitle(f"Time: t=0")
        if one_cbar:
            if plt_shape[0] > plt_shape[1]:
                fig.colorbar(
                    ims[0],
                    ax=axs,
                    orientation="vertical",
                    location="right",
                    label="Excitation(x, t)",
                )
            else:
                fig.colorbar(
                    ims[0],
                    ax=axs,
                    orientation="horizontal",
                    location="bottom",
                    label="Excitation(x, t)",
                )

        def animation_function(i):
            k = i * int(t.shape[0] / (fps * T_out))
            tk = t[k]
            for i, im in enumerate(ims):
                im.set_array(animation_data[i][k])
            fig.suptitle("Time: t=%i" % tk)
            return ims

        anim = animation.FuncAnimation(
            fig,
            animation_function,
            frames=round(T_out * fps),
            interval=1000 / fps,  # in ms
        )

        if figure_name is None:
            figure_name = "".join([f"{name}_" for name in names])[:-1]

        logging.info(f"Writing animation to {Path}{figure_name}.mp4")
        if prog_feedback:
            anim.save(
                f"{Path}{figure_name}.mp4",
                fps=fps,
                extra_args=["-vcodec", "libx264"],
                progress_callback=progress_feedback.printProgressBar,
            )
        else:
            print(f"Animation safed at: {Path}{figure_name}.mp4")
            anim.save(
                f"{Path}{figure_name}.mp4",
                fps=fps,
                extra_args=["-vcodec", "libx264"],
            )
        logging.info(f"Animation safed at: {Path}{figure_name}.mp4")
        plt.close()

    # I am not sure whether this function still works!
    @staticmethod
    def produce_series_of_snapshots(
        animation_data_u,
        frame_nrs,
        T_tot,
        bounds=None,
        format="1x4_only_u",
        animation_data_w=None,
        with_heat=None,
        name=None,
    ):
        """
        The function produces series of snapshots (heatmaps) for different possibilities
        of numbers of images, formats, and possibly from different data sets.

        Args:
            animation_data: Data of excitation over time. First dimension is temporal (three dim np array)
            frame_nrs: A list of the numbers of frams which shall be included in the saved figure (array)
            T_tot: Physical temporal length of animation (float)
            bounds: bounds of the range in heatmap (float array of shape (2,))
            format: Specification of the format (Str)
            name: name of file, and title of plot (Str)

        Output:
            image

        Warning:
            Not maintained! And very old.

        """
        raise NotImplementedError(
            "This function is not maintained and should not be used!"
        )
        frame_nrs = np.array(frame_nrs)
        t = T_tot * (frame_nrs / animation_data_u.shape[0])
        if format == "1x4_only_u":
            if bounds is None:
                bounds = [np.min(animation_data_u), np.max(animation_data_u)]
            elif np.array(bounds).shape != (2,):
                logging.info(
                    "Warning: Shape of 'bounds' needs to be (2,) but is {bounds.shape} minimum and maximum of data are chosen as minimum."
                )
            fig, ax = plt.subplots(ncols=4, nrows=1)
            im = ax[0].imshow(
                animation_data_u[frame_nrs[0]], vmin=bounds[0], vmax=bounds[1]
            )
            ax[0].set_title("t=%i" % t[0])
            ax[1].imshow(animation_data_u[frame_nrs[1]], vmin=bounds[0], vmax=bounds[1])
            ax[1].set_title("t=%i" % t[1])
            ax[1].axes.yaxis.set_ticklabels([])
            ax[2].imshow(animation_data_u[frame_nrs[2]], vmin=bounds[0], vmax=bounds[1])
            ax[2].set_title("t=%i" % t[2])
            ax[2].axes.yaxis.set_ticklabels([])
            ax[3].imshow(animation_data_u[frame_nrs[3]], vmin=bounds[0], vmax=bounds[1])
            ax[3].set_title("t=%i" % t[3])
            ax[3].axes.yaxis.set_ticklabels([])
            # cbar_ax = fig.add_axes([0.15, 0.2, 0.7, 0.05])
            # fig.colorbar(im, cax=cbar_ax, orientation = 'horizontal')
        if format == "2x4_with_w":
            if bounds is None:
                bounds = [
                    np.min(animation_data_u),
                    np.max(animation_data_u),
                    np.min(animation_data_w),
                    np.max(animation_data_w),
                ]
            elif np.array(bounds).shape != (4,):
                logging.info(
                    f"Warning: Shape of 'bounds' needs to be (4,) but is {bounds.shape} minimum and maximum of data are chosen as minimum."
                )
                bounds = [
                    np.min(animation_data_u),
                    np.max(animation_data_u),
                    np.min(animation_data_w),
                    np.max(animation_data_w),
                ]
            fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(10, 5))
            gs1 = gridspec.GridSpec(4, 2, wspace=0.005, hspace=0)
            # gs1.update(wspace=0.005, hspace=0.005)
            axs[0, 0].imshow(
                animation_data_u[frame_nrs[0]], vmin=bounds[0], vmax=bounds[1]
            )
            axs[0, 0].set_title("t=%i" % t[0])
            axs[0, 0].set_ylabel("Excitation of u")
            axs[0, 0].axes.xaxis.set_ticks([0, 100, 200, 300])
            axs[0, 0].axes.yaxis.set_ticks([0, 100, 200, 300])
            axs[0, 0].axes.xaxis.set_ticklabels([])
            axs[0, 1].imshow(
                animation_data_u[frame_nrs[1]], vmin=bounds[0], vmax=bounds[1]
            )
            axs[0, 1].set_title("t=%i" % t[1])
            axs[0, 1].axes.xaxis.set_ticks([0, 100, 200, 300])
            axs[0, 1].axes.yaxis.set_ticks([0, 100, 200, 300])
            axs[0, 1].axes.yaxis.set_ticklabels([])
            axs[0, 1].axes.xaxis.set_ticklabels([])
            axs[0, 2].imshow(
                animation_data_u[frame_nrs[2]], vmin=bounds[0], vmax=bounds[1]
            )
            axs[0, 2].set_title("t=%i" % t[2])
            axs[0, 2].axes.xaxis.set_ticks([0, 100, 200, 300])
            axs[0, 2].axes.yaxis.set_ticks([0, 100, 200, 300])
            axs[0, 2].axes.yaxis.set_ticklabels([])
            axs[0, 2].axes.xaxis.set_ticklabels([])
            im = axs[0, 3].imshow(
                animation_data_u[frame_nrs[3]], vmin=bounds[0], vmax=bounds[1]
            )
            axs[0, 3].set_title("t=%i" % t[3])
            axs[0, 3].axes.xaxis.set_ticks([0, 100, 200, 300])
            axs[0, 3].axes.yaxis.set_ticks([0, 100, 200, 300])
            axs[0, 3].axes.yaxis.set_ticklabels([])
            axs[0, 3].axes.xaxis.set_ticklabels([])
            # plt.colorbar(axs[0,3].imshow(animation_data_u[frame_nrs[3]], vmin = bounds[0], vmax = bounds[1]),cax=axs[0,4])
            axs[1, 0].imshow(
                animation_data_w[frame_nrs[0]], vmin=bounds[2], vmax=bounds[3]
            )
            axs[1, 0].set_ylabel("Excitation of w")
            axs[1, 0].axes.xaxis.set_ticks([0, 100, 200, 300])
            axs[1, 0].axes.yaxis.set_ticks([0, 100, 200, 300])
            axs[1, 1].imshow(
                animation_data_w[frame_nrs[1]], vmin=bounds[2], vmax=bounds[3]
            )
            axs[1, 1].axes.yaxis.set_ticklabels([])
            axs[1, 1].axes.xaxis.set_ticks([0, 100, 200, 300])
            axs[1, 1].axes.yaxis.set_ticks([0, 100, 200, 300])
            axs[1, 2].imshow(
                animation_data_w[frame_nrs[2]], vmin=bounds[2], vmax=bounds[3]
            )
            axs[1, 2].axes.xaxis.set_ticks([0, 100, 200, 300])
            axs[1, 2].axes.yaxis.set_ticks([0, 100, 200, 300])
            axs[1, 2].axes.yaxis.set_ticklabels([])
            im2 = axs[1, 3].imshow(
                animation_data_w[frame_nrs[3]], vmin=bounds[2], vmax=bounds[3]
            )
            axs[1, 3].axes.yaxis.set_ticklabels([])
            axs[1, 3].axes.xaxis.set_ticks([0, 100, 200, 300])
            axs[1, 3].axes.yaxis.set_ticks([0, 100, 200, 300])
            # plt.colorbar(axs[1,3].imshow(animation_data_u[frame_nrs[3]], vmin = bounds[2], vmax = bounds[3]),cax=axs[1,4])

            if with_heat != None:
                fig.subplots_adjust(right=0.85)
                cbar_ax = fig.add_axes([0.9, 0.6, 0.025, 0.25])
                fig.colorbar(im, cax=cbar_ax)
                cbar_ax = fig.add_axes([0.9, 0.2, 0.025, 0.25])
                fig.colorbar(im2, cax=cbar_ax)
        if name is not None:
            plt.savefig(name + ".pdf")
        else:
            plt.show()
