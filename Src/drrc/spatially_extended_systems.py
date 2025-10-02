"""
Solver for spatially extended systems.

.. codeauthor:: Luk Fleddermann
"""
import numpy as np

from .tools.general_methods import progress_feedback


class LocalModel:
    """
    Local Model of Exitation.
    """

    def __init__(self, Pardict) -> None:
        """
        The init function creates the environment for the different models of the electric excitation,
        i.e. it sets parameters for differential equation and a propper Grid-size (Physical and numerical) for the animation.

        Args:
            Pardict: Parameterfile
        """
        # Pardict = Pardict["LocalModel"]

        self.a = Pardict["LocalModel"]["Parameter"]["a"]
        self.b = Pardict["LocalModel"]["Parameter"]["b"]
        self.eps = Pardict["LocalModel"]["Parameter"]["eps"]

        valid_names = ["FitzHugh-Nagumo", "Aliev-Panfilov"]

        if Pardict["Name"] not in valid_names:
            raise ValueError(
                f"Pardict['Name'] needs to be {valid_names} but is {Pardict['Name']}."
            )
        else:
            if Pardict["Name"] == "FitzHugh-Nagumo":
                self.name = "FitzHugh-Nagumo"
                self.apply = self.fitzhugh_nagumo_local
            elif Pardict["Name"] == "Aliev-Panfilov":
                self.name = "Aliev-Panfilov"
                self.mu1 = Pardict["LocalModel"]["Parameter"]["mu1"]
                self.mu2 = Pardict["LocalModel"]["Parameter"]["mu2"]
                self.k = Pardict["LocalModel"]["Parameter"]["k"]
                self.apply = self.aliev_panfilov_local

    def fitzhugh_nagumo_local(self, vars):
        r"""
        The local FitzHugh-Nagumo model follows

        .. math::

            \partial_t u(x,y,t) &= au(u-b)(1-u)-w\,,\\
            \partial_t w(x,y,t) &= -\varepsilon(u-w)\,,

        which is extended to the spatially extended FitzHugh-Nagumo model by adding a diffusion term to the u variable.

        Args:
            vars (np.ndarray): excitation of variables :code:`[u,w] = [vars[0], vars[1]]`
        
        Return:
            np.ndarray: change in excitation of u and w following the fitz_nagumo model.
            If :class:`LocalModel` is of wrong type, returns -1.

        Notes:
            The class needs to be initialized, with :code:`model='fitzhugh_nagumo`,
            for the function to work.
        """
        dvars = np.zeros(vars.shape)
        dvars[0] = self.a * vars[0] * (vars[0] - self.b) * (1 - vars[0]) - vars[1]
        dvars[1] = self.eps * (vars[0] - vars[1])
        return dvars

    def aliev_panfilov_local(self, vars):
        r"""
        The local Aliev-Panfilov model follows

        .. math::

            \partial_t u(x,y,t) &= ku(u-a)(1-u)-uw\,,\\
            \partial_t w(x,y,t) &= \left(\varepsilon+\frac{\mu_1 w}{\mu_2 +u}\right)\cdot(-w-ku(u-b-1))\,,
        
        which is extended to the spatially extended Aliev-Panfilov model by adding a diffusion term to the u variable.

        Args:
            vars (np.ndarray): excitation of variables
        Return:
            np.ndarray: change in excitation of u and w following the fitz_nagumo model.
            If class.model is of wrong type, returns -1.
        Notes:
            The class needs to be initialized, with :code:`model='aliev_panfilov`,
            for the function to work.
        """
        dvars = np.zeros(vars.shape)
        dvars[0] = (
            self.k * vars[0] * (1 - vars[0]) * (vars[0] - self.a) - vars[0] * vars[1]
        )
        dvars[1] = (self.eps + self.mu1 * vars[1] / (self.mu2 + vars[0])) * (
            -vars[1] - self.k * vars[0] * (vars[0] - self.b - 1)
        )
        return dvars


class Diffusion:
    """
    Diffusion model, different accuracy (5 point/ 9 point stencil).
    May be extended to other methods of implementing diffusion.
    """

    def __init__(self, Pardict) -> None:
        """
        Args:
            model (str):
                Specification of the model of partial differential eq. one wants to use
            stencil (str):
                Specification of the stencil used in the calculation of the Laplacian
            stable (bool):
                Choice whether stable or chaotic spiral waves are produced with the
                Aliev-Panfilov model
        """
        valid_stencils = ["NinePoint", "FivePoint"]
        if Pardict["Diffusion"]["stencil"] not in valid_stencils:
            raise ValueError(
                f"'Diffusion Stencil' needs to be {valid_stencils} but is {Pardict['Diffusion']['Stencil']}."
            )
        else:
            self.d = Pardict["Diffusion"]["constant"]
            self.dx = Pardict["SpatialParameter"]["dx"]
            if Pardict["Diffusion"]["stencil"] == "FivePoint":
                self.apply = self.apply_5stencil
            if Pardict["Diffusion"]["stencil"] == "NinePoint":
                self.apply = self.apply_9stencil

    def apply_5stencil(self, u):
        """
        Args:
            u: excitation of grid.
        Return:
            laplacian: Two dimensional array of discrete Laplacian of the excitation of
            u, using the five-point stencil.
        Notes:
            The outer layer of u needs to be a ghost layer s.t. no flux boundary
            conditions are inforced.
            Without ghost layer periodic boundary conditions are used.
        """
        laplacian = -4 * u
        laplacian += np.roll(u, 1, axis=0)
        laplacian += np.roll(u, -1, axis=0)
        laplacian += np.roll(u, 1, axis=1)
        laplacian += np.roll(u, -1, axis=1)
        return self.d * laplacian / (self.dx**2)

    def apply_9stencil(self, u):
        """
        Args:
            u:
                excitation of grid.
        Return:
            laplacian:
                Two dimensional array of discrete Laplacian of the excitation of u,
                using the five-point stencil.
        Notes:
            The outer layer of u needs to be a ghost layer s.t. no flux boundary
            conditions are inforced.
            Without ghost layer periodic boundary conditions are used.
        """
        laplacian = -20 * u
        laplacian += 4 * np.roll(u, 1, axis=0)
        laplacian += 4 * np.roll(u, -1, axis=0)
        laplacian += 4 * np.roll(u, 1, axis=1)
        laplacian += 4 * np.roll(u, -1, axis=1)
        laplacian += np.roll(np.roll(u, 1, axis=0), 1, axis=1)
        laplacian += np.roll(np.roll(u, 1, axis=0), -1, axis=1)
        laplacian += np.roll(np.roll(u, -1, axis=0), 1, axis=1)
        laplacian += np.roll(np.roll(u, -1, axis=0), -1, axis=1)
        return self.d / 6 * laplacian / (self.dx**2)


class BoundaryCondition:
    """
    Different Typs of Boundary conditions. So far only 'no_flux' has a proper meaning.
    """

    def __init__(self, Pardict) -> None:
        """
        The init function creates the environment for the different models of the
        electric excitation, i.e. it sets parameters for differential equation and a
        proper Grid-size (Physical and numerical) for the animation.

        Args:
            model (str):
                Specification of the model of partial differential eq. one wants to use
            stencil (str):
                Specification of the stencil used in the calculation of the Laplacian
            stable (bool):
                Choice wether stable or chaotic spiral waves are produced with the
                Aliev-Panfilov model
        """
        if Pardict["BoundaryCondition"]["type"] not in [
            "NoFlux",
            "SetBoundary",
            "Periodic",
        ]:
            raise KeyError(
                "Error: Initialisation incomplete. 'Pardict['BoundaryCondition']' needs"
                "to be 'aliev_panfilov' or 'fitzhugh_nagumo' but is",
                Pardict["BoundaryCondition"],
                ".",
            )
        else:
            if Pardict["BoundaryCondition"]["type"] == "NoFlux":
                self.name = "NoFlux"
                self.apply = self.no_flux
            elif Pardict["BoundaryCondition"]["type"] == "SetBoundary":
                self.name = "SetBoundary"
                print("Loading Boundary.")
                self.bound_var1 = np.load(Pardict["BoundaryCondition"]["Path"])["vars"][
                    :, 0
                ]
                self.apply = self.set_bound_u

    def no_flux(self, vars, itterator=None):
        """
        Args:
            u: excitation of grid.
        Return:
            u: excitation of grid with no flux boundary condition enforced for
            5-point stencil.
        Notes:
            The outer layer is turned in a ghost layer with no physical meaning.
            The corners of the grid (i.e. u[0,0]) are unimportant for the calculations
            of the five point stencil, hence orientation of the four executed steps does
            not matter.
        """
        vars[0, -1, :] = vars[0, -2, :]
        vars[0, 0, :] = vars[0, 1, :]
        vars[0, :, -1] = vars[0, :, -2]
        vars[0, :, 0] = vars[0, :, 1]
        return vars

    def set_bound_u(self, vars, itterator):
        vars[0, 0] = self.bound_var1[itterator, 0]
        vars[0, -1] = self.bound_var1[itterator, -1]
        vars[0, :, 0] = self.bound_var1[itterator, :, 0]
        vars[0, :, -1] = self.bound_var1[itterator, :, -1]
        return vars


class IntegrationMethods:
    """
    Methods of temporal integraion of class 'LocalModel' coupled by class 'Diffusion'.
    """

    @staticmethod
    def exp_euler_itt(
        dif: Diffusion,
        lm: LocalModel,
        bc: BoundaryCondition,
        Pardict: dict,
        seed: int | None = None,
    ):
        """
        Iterative application of explicit Euler method.

        Args:
            dif: Diffusion Model
            lm: Local Model
            bc: Boundary Conditions
            Pardict: Parameter File
            seed: seed of random initial condition

        Return:
            vars: Last time step of integration
        """
        T = Pardict["TemporalParameter"]["Transient"]
        dt = Pardict["TemporalParameter"]["integration_dt"]
        vars = IntegrationMethods._get_initial(Pardict["SystemParameters"], seed=seed)
        vars = bc.apply(vars, 0)
        for n_t, t in enumerate(np.arange(dt, T, dt)):
            vars += dt * lm.apply(vars)
            vars[0] += dt * dif.apply(vars[0])
            vars = bc.apply(vars, itterator=n_t)
            if (n_t) % 1000 == 0 and n_t > 0:
                progress_feedback.printProgressBar(
                    int((n_t)), int(T // dt), name="Integration without saving:"
                )
        return vars

    @staticmethod
    def exp_euler_itt_with_save(
        dif: Diffusion,
        lm: LocalModel,
        bc: BoundaryCondition,
        Pardict: dict,
        vars0: np.ndarray | None = None,
        seed: int | None = None,
        prog_feetback: bool = True,
    ):
        """
        Iterative application of explicit euler method.

        Args:
            dif: Diffusion Model
            lm: Local Model
            bc: Boundary Conditions
            Pardict: Parameter File
            vars0: Initial Condition
            seed: seed of random initial condition

        Return:
            vars: Last time step of integration
        """

        T = Pardict["TemporalParameter"]["T"]
        dt = Pardict["TemporalParameter"]["integration_dt"]

        if not np.isclose(1000 * Pardict["TemporalParameter"]["dt"] % (1000 * dt), 0):
            print(
                dt,
                Pardict["TemporalParameter"]["dt"],
                Pardict["TemporalParameter"]["dt"] % dt,
            )
            raise TypeError("Saving Resolution needs to be multiple of temporal step.")
        else:
            save_each = int(Pardict["TemporalParameter"]["dt"] / dt)
        vars = IntegrationMethods._get_initial(Pardict, vars0, seed=seed)

        t_save = np.arange(0, T, dt * save_each)

        saving_shape = tuple([t_save.shape[0], vars.shape[0]]) + tuple(
            np.array(vars.shape[1:]) - 2
        )
        vars_save = np.zeros(shape=saving_shape, dtype="float32")
        vars = bc.apply(vars, 0)
        for n_t, t in enumerate(np.arange(dt, T, dt)):
            vars += dt * lm.apply(vars)
            vars[0] += dt * dif.apply(vars[0])
            vars = bc.apply(vars, itterator=n_t)
            if (n_t) % save_each == 0 and n_t > 0:
                vars_save[int((n_t) / save_each)] = vars[
                    :, 1:-1, 1:-1
                ]  # removing boundary
                if prog_feetback:
                    progress_feedback.printProgressBar(
                        int((n_t) / save_each),
                        t_save.shape[0],
                        name="Integration with saving:   ",
                    )
        return vars_save, t_save

    @staticmethod
    def _get_initial(
        Pardict: dict, vars0: np.ndarray | None = None, seed: int | None = None
    ) -> np.ndarray:
        """
        Creates initial condition (array) from string in dictionary.

        Args:
            Pardict (dict): initial condition
            vars0 (np.ndarray): initial condition
            seed (int): seed of random initial condition

        Return:
            np.ndarray: initial condition

        Notes:
            Pardict needs to have keys 'LocalModel', 'SpatialParameter' and
            'initial_condition'
        """
        if isinstance(vars0, np.ndarray):
            return vars0
        else:
            vars = np.zeros(
                (
                    Pardict["LocalModel"]["NrVariables"],
                    Pardict["SpatialParameter"]["spatial_dimension"] + 2,
                    Pardict["SpatialParameter"]["spatial_dimension"] + 2,
                )
            )
            if Pardict["initial_condition"] == "AllZero":
                pass
            elif (
                Pardict["initial_condition"] == "InitiateSpiral"
                and Pardict["Name"] == "Aliev-Panfilov"
            ):
                vars[
                    0,
                    1 : round(
                        2 * (Pardict["SpatialParameter"]["spatial_dimension"] + 2) / 5
                    ),
                    :,
                ] = 1
                vars[
                    1,
                    1 : round(
                        2 * (Pardict["SpatialParameter"]["spatial_dimension"] + 2) / 5
                    ),
                    :,
                ] = 0.5
                vars[
                    1,
                    round(
                        2 * (Pardict["SpatialParameter"]["spatial_dimension"] + 2) / 5
                    ) : (Pardict["SpatialParameter"]["spatial_dimension"] + 2),
                    round((Pardict["SpatialParameter"]["spatial_dimension"] + 2) / 2) :,
                ] = 1
            elif (
                Pardict["initial_condition"] == "InitiateSpiral"
                and Pardict["Name"] == "FitzHugh-Nagumo"
            ):
                vars[
                    0,
                    1 : round(
                        2 * (Pardict["SpatialParameter"]["spatial_dimension"] + 2) / 5
                    ),
                    :,
                ] = 1
                vars[
                    1,
                    round(
                        2 * (Pardict["SpatialParameter"]["spatial_dimension"] + 2) / 5
                    ) : (Pardict["SpatialParameter"]["spatial_dimension"] + 2),
                    : round((Pardict["SpatialParameter"]["spatial_dimension"] + 2) / 2),
                ] = 0.2
            elif (
                Pardict["initial_condition"] == "RandomChaos"
                and Pardict["Name"] == "Aliev-Panfilov"
            ):
                sd = Pardict["SpatialParameter"]["spatial_dimension"]
                vars[0, 0 : int((sd + 2) / 2)] = 1
                np.random.seed(seed)
                vars[1] = np.random.rand(*vars[1].shape)
                vars[1, int((sd + 2) / 2) :, : int(sd / 2)] = 2.5
            return vars
