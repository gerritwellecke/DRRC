"""
Generates training data for a Kuramoto-Sivashinsky system.

.. codeauthor:: Gerrit Wellecke
"""
import argparse
import os
import sys
from pathlib import Path

import git
import matplotlib.pyplot as plt
import numpy as np
import pde
import yaml
from matplotlib import cm
from tqdm import tqdm

from .config import Config


class KuramotoSivashinsky:
    r"""
    The Kuramoto-Sivashinsky-equation as

    .. math::
        \partial_t u(x,t) = -\frac{1}{2} \nabla\left[ u^2(x,t)\right] - \nabla^2 u(x,t)
            - \nu \nabla^4 u(x,t)\,,

    where :math:`u` is a one-dimensional field.
    """

    def __init__(
        self,
        config: Config,
        method: str,
        nu: float,
        L: float,
        nx: int,
        dt: float,
        t0: float = 0,
        tN: float = 0,
        task_id: int = 0,
        Filepath: str = "Data/1D_KuramotoSivashinsky/",
        cluster_save: bool = False,
        **kwargs,
    ):
        """
        Todo:
            Update documentation!
        Args:
            config (Config):
                configuration YAML object referring to the KS system
            method (str):
                method of integration.
            nu (float):
                prefactor of 4th derivative in KS equation
            L (float):
                domain length, i.e. domain is [0, L]
            nx (int):
                number of gridpoints in spatial domain
            dt (float):
                time step size
            t0 (float):
                    time before is integrated but discarded
            tN (float):
                final time point
            task_id (int):
                SGE_TASK_ID of the current execution
            cluster_save (bool):
                Set to true if you wish to save data with an absolute path.
                In this case Jobscript-Datadir is interpreted absolute, which is useful
                when you want to use the cluster's data server.


        Example:
            This best used by dictionary unpacking, e.g.

            .. code:: python

                KuramotoSivashinsky(
                    config=conf,
                    **conf["System"]["Parameters"],
                    **conf.param_scan_list(task_id),
                    task_id=task_id
                )

        Note:
            The :class:`drrc.config.Config` for this look as follows:

            .. code:: YAML

                System:
                  Name: "KuramotoSivashinsky"
                  Parameters:
                    # nu: 1
                    L: 60
                    nx: 128
                    t0: 0
                    tN: 30000
                    dt: 0.25
                  Method: "spectral"  # spectral or py-pde

                Saving:
                  # this must be absolute paths with respect to the repository root
                  Name: "KS_t"
                  PlotPath: "Figures/"

                Jobscript:
                  Name: "Example-2023"
                  Datadir: "~"  # this has to be an absolute path on the executing system
                  Cores: 4

                ParamScan:
                  nu: [0.1, 1.1, 0.2]
        """
        # dict of all simulation parameters
        self.nu = nu
        self.L = L
        self.nx = nx
        self.dt = dt
        self.task_id = task_id

        # number of total time steps
        self.t0 = t0
        self.tN = tN
        self.nt = int((self.tN) / self.dt) + 1  # total number of integration steps
        self.nt0 = (
            int((self.t0) / self.dt) + 1
        )  # number of integration steps which are discarded

        # time series as np.ndarray
        self.u = np.empty((self.nt0 + self.nt, self.nx))
        # integration method as str, ["spectral", "py-pde"]
        # TODO fix the next line -- this is super inconsistent!
        self.method = method
        # full path to data file
        self.outfile = str(config.get_git_root() / Filepath)
        # directory for plots
        self.plotdir = str(
            config.get_git_root() / "Figures/1D_KuramotoSivashinsky/Plot.pdf"
        )
        self.gitroot = str(config.get_git_root()) + "/"

    def generate_timeseries(self, **kwargs):
        """Integrate KS system with the integrator specified in the config YAML

        Args:
            **kwargs: Can include and overwrite

                t0 (float):
                    time before is integrated but discarded
                tN (float):
                    final time point
        """

        for arg in kwargs:
            if arg == "t0":
                self.t0 = kwargs["t0"]
            if arg == "tN":
                self.tN = kwargs["tN"]
            self.nt = int((self.tN) / self.dt) + 1  # total number of integration steps
            self.nt0 = (
                int((self.t0) / self.dt) + 1
            )  # number of integration steps which are discarded

        if self.method == "spectral":
            self.generate_timeseries_spectral()
        elif self.method == "py-pde":
            self.generate_timeseries_pypde()
        else:
            raise ValueError("Must enter a valid integration method.")

    def generate_timeseries_spectral(self):
        """Integrate the KS system using a spectral method"""
        # wave number mesh for real FFT
        k = np.arange(0, self.nx / 2 + 1, 1)

        t = np.linspace(start=self.t0, stop=self.tN, num=self.nt)
        x = np.linspace(start=0, stop=self.L, num=self.nx)
        u = np.empty((self.nx, self.nt))

        # indices of fourier space of rfft (# of datapts/2 + 1 for zero)
        fft_indices = int(self.nx / 2) + 1
        # solution mesh in Fourier space
        u_hat = np.ones((fft_indices, self.nt), dtype=complex)
        u_hat2 = np.ones((fft_indices, self.nt), dtype=complex)

        # initial condition
        u0 = np.random.rand() * np.cos(
            (2 * np.pi * x) / self.L
        ) + np.random.rand() * np.cos((4 * np.pi * x) / self.L)

        # Fourier transform of initial condition
        u0_hat = (1 / self.nx) * np.fft.rfft(u0)
        u0_hat2 = (1 / self.nx) * np.fft.rfft(u0**2)

        # set initial condition in real and Fourier mesh
        u[:, 0] = u0
        u_hat[:, 0] = u0_hat
        u_hat2[:, 0] = u0_hat2

        # Fourier Transform of the linear operator
        FL = (((2 * np.pi) / self.L) * k) ** 2 - self.nu * (
            ((2 * np.pi) / self.L) * k
        ) ** 4
        # Fourier Transform of the non-linear operator
        FN = -(1 / 2) * ((1j) * ((2 * np.pi) / self.L) * k)

        # resolve PDE in Fourier space
        for j in tqdm(range(0, self.nt - 1), desc="Integration (spectral)"):
            # set u(k,t)
            uhat_current = u_hat[:, j]
            uhat_current2 = u_hat2[:, j]

            # set u(k, t-dt)
            if j == 0:
                # uhat_last = u_hat[:, 0]
                uhat_last2 = u_hat2[:, 0]
            else:
                # uhat_last = u_hat[:, j-1]
                uhat_last2 = u_hat2[:, j - 1]

            # compute solution in Fourier space through a finite difference method
            # Cranck-Nicholson
            crTerm = (1 + (self.dt / 2) * FL) * uhat_current
            # Adam Bashforth
            abTerm = (3 / 2 * FN * uhat_current2 - 1 / 2 * FN * uhat_last2) * self.dt
            # full PDE
            u_hat[:, j + 1] = (1 / (1 - self.dt / 2 * FL)) * (crTerm + abTerm)

            # compute solution in real space, i.e. u(x,t+dt)
            u[:, j + 1] = self.nx * np.fft.irfft(u_hat[:, j + 1])

            # compute the Fourier transform of u^2
            u_hat2[:, j + 1] = (1 / self.nx) * np.fft.rfft(u[:, j + 1] ** 2)

        self.u = u[:, self.nt0 :]

    def generate_timeseries_pypde(self):
        """Integrate the KS system using :code:`py-pde`.

        Note:
            :code:`py-pde` is much better tested than the code of
            :func:`generate_timeseries_spectral`.
            Also it solves the system in real space.
            However, this comes at a much slower speed!
        """
        # make 1D grid
        grid = pde.CartesianGrid([(0, self.L)], [self.nx], periodic=True)

        # initial condition: random field
        state = pde.ScalarField.from_expression(
            grid, f"cos((2 * pi * x) / {self.L}) + 0.1 * cos((4 * pi * x) / {self.L})"
        )

        # define Kuramoto-Sivashinksky PDE
        eq = pde.PDE({"u": "-u * d_dx(u) - laplace(u + laplace(u))"})

        # solve the system
        storage = pde.MemoryStorage()
        result = eq.solve(
            state,
            t_range=self.tN,
            dt=self.dt,
            adaptive=True,
            tracker=["progress", storage.tracker(self.dt)],
        )

        self.u = np.array(storage.data).T[:, self.nt0 :]

    def save(self, filename: str | None = None):
        """Save time series to .npy file

        Args:
            filename (str):
                If specified, this is the resulting file without the extension.
                Otherwise the filename from the config YAML is used.
        """
        if filename is None:
            np.save(f"{self.outfile}KS_Data", self.u.T)
            print("Saving Data at " + f"{self.outfile}")
        else:
            np.save(f"{filename}", self.u.T)
            print("Saving Data at " + f"{filename}")

    def plot_conservation(self):
        """Plot field conversation for quick sanity checks

        Attention:
            This method does not yet allow for writing the resulting plot to file.
        """
        t = np.linspace(start=self.t0, stop=self.tN, num=self.nt)

        fig = plt.figure(figsize=(4, 4 / 1.618))
        plt.plot(t, np.sum(self.u, axis=0))
        plt.xlabel("$t$")
        plt.ylabel("$\\langle u(x,t) \\rangle_x$")
        plt.tight_layout()
        plt.show()

    def plot_kymograph(self, SavingPath: str | None = None):
        """Plot kymograph of the system.

        Args:
            save (str):
                If set the result are not shown but instead saved to a png at the given location.
        """
        fig, ax = plt.subplots(1, 2, figsize=(10, 8), tight_layout=True)
        cs = ax[0].imshow(
            self.u.T, cmap=cm.get_cmap("jet"), aspect="auto", origin="lower"
        )
        cs = ax[1].imshow(
            self.u.T[:1000], cmap=cm.get_cmap("jet"), aspect="auto", origin="lower"
        )

        fig.colorbar(cs)
        ax[0].set_xlabel("x")
        ax[0].set_ylabel("t")
        fig.suptitle(f"Kuramoto-Sivashinsky: L = {self.L}, nu = {self.nu}")

        if SavingPath is not None:
            try:
                os.mkdir(self.gitroot + SavingPath[:31])
                print("Created Outputfolder:", self.gitroot + SavingPath[:31])
            except FileExistsError:
                pass
            print("Saving Figure at " + self.gitroot + SavingPath + ".png")
            plt.savefig(self.gitroot + SavingPath + ".png")
            plt.close()
        else:
            plt.show()
