"""
This module contains the implementation of the ParallelReservoirsBase class, three coresponding subclasses ParallelReservoirs, ParallelReservoirsFFT and ParallelReservoirsPCA and the ParallelReservoirsArguments dataclass.

The Subclasses are used to predict high dimensional (spatially extended) systems using multiple Reservoir Computers in parallel:
    1. The ParallelReservoirs class is used for high dimensional systems, without a dimension reduction.
    2. The ParallelReservoirsFFT class is used for high dimensional systems, using only a selection of fft modes as a dimension reduction.
    3. The ParallelReservoirsPCA class is used for high dimensional systems, using only a selection of pca modes as a dimension reduction.

The base class :code:`ParallelReservoirsBase` contains everything for parallel reservoir applications that predicts high dimensional (spatially extended) systems using multiple Reservoir Computers in parallel.
The classes use a class instance of :code:`ParallelReservoirsArguments`, which is a dataclass that represents the parameters for configuring the (parallel-) reservoirs.

Author: Luk Fleddermann, Gerrit Wellecke
Date: 13.06.2024
"""

import logging as log
import os.path as paths
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields

import numpy as np
from sklearn.decomposition import PCA

from drrc.config import Config
from drrc.reservoircomputer import ReservoirComputer
from drrc.tools.logger_config import drrc_logger as logger

# NOTE: what ParallelReservoirsX are supposed to do:
# - Implement the use of multi-reservoirs
# - Contain a list of those reservoirs
# - Train a reservoir for each of the subdomains or alternatively train one reservoir on
#   all subdomains
# - Handle the training & output data w.r.t. linear transformations
#   (currently PCA & FFT) and boundary conditions (through ghost cells)

# Coding
# TODO: JIT the _boundary_condition and _evaluation_one_step, this is used in every prediction step
# TODO: Maybe add tqdm to Training loop of reservoirs, decide what we want there some times the number is 1-4 sometimes up to at least 2000

# Testing and documentation
# TODO: write wrong parameter usage function. For instance, ParallelReservoirs should not have dimensionreduction
# TODO: Insert wrong arguments checking for all non-internal functions (init, train, itterative_predict, reservoir_transient)
# TODO: remove class variables from documentation

# get logging levels
loglevel_debug_parallelreservoirs = log.getLevelName("DEBUG_PRC")
loglevel_debug_DRRC = log.getLevelName("DEBUG_DRRC")

loglevel_info_single_run = log.getLevelName("INFO_1RUN")
loglevel_info_multiple_run = log.getLevelName("INFO_nRUN")


@dataclass
class ParallelReservoirsArguments:
    """Parallel-reservoir parameters.

    Args:
        adjacency_degree (int):
            average degree of the adjacency matrix
        adjacency_dense (bool):
            whether the adjacency matrix is sparse (False) or dense (True)
        adjacency_spectralradius (float):
            spectral radius, largest eigenvalue of the adjacency matrix
        reservoir_leakage (float):
            leakage the strength of memory a reservoir state remembers old
            excitations with (0.0 only driven from new data no memory, 1.0 no update
            of rs)
        reservoir_nodes (int):
            number of nodes in each of the parallel reservoirs
        input_scaling (float):
            input scaling the maximal absolute value of entries in the input matrix
        input_bias (float):
            scaling of the bias strength double the maximal absolute value of the
            bias input to a reservoir node, None defaults to inscale
        spatial_shape (tuple[int, ...]):
            shape of the input data, without boundary condition (for ex. :code:`(128,)` in 1D case, :code:`(128,128)` in 2D case)
        system_variables (int):
            number of variables in the system (data for one time step is of shape :code:`(system_variables, *spatial_shape)`).
        boundary_condition (str):
            the type of boundary condition to apply to the input data
        parallelreservoirs_grid_shape (tuple[int, ...]):
            the amount of reservoirs per dimension that should be used together as a
            multi-reservoir (for ex. [2,1] in 2D case for 2 reservoirs in
            x direction)
        parallelreservoirs_ghosts (int):
            number of variables that a reservoir sees from outside the region where
            its predicting for sync.
        dimensionreduction_fraction (float):
            fraction of variables that actually enters the reservoir after
            dimension reduction
        training_includeinput (bool):
            whether to also fit the input signal for predicting the next timestep
        training_regularization (float):
            regularization strength for the ridge regression
        training_output_bias (float):
            scaling of the output bias
        identical_inputmatrix (bool):
            whether we use different input matrices for each reservoir
        identical_adjacency (bool):
            whether we use different adjacency matrices for each reservoir
        identical_outputmatrix (bool | str | tuple[int, ...]):
            whether we train each domain with a separate reservoir (False) or one
            reservoir on all domains ('combine_data') or one reservoir on one domain (tuple of indices)
    """

    adjacency_degree: int
    adjacency_dense: bool
    adjacency_spectralradius: float
    reservoir_leakage: float
    reservoir_nodes: int
    input_scaling: float
    input_bias: float
    spatial_shape: tuple[int, ...]
    system_variables: int
    boundary_condition: str
    parallelreservoirs_grid_shape: tuple[int, ...]
    parallelreservoirs_ghosts: int
    dimensionreduction_fraction: float
    training_includeinput: bool
    training_regularization: float

    training_output_bias: float
    identical_inputmatrix: bool
    identical_adjacencymatrix: bool
    identical_outputmatrix: bool | str | tuple[int, ...]

    # maybe deprecated, not used yet
    # training_overwrite: bool
    # dtype: str
    # deprecated_trainonly: bool

    @classmethod
    def from_config(cls, conf: Config, job_idx: int, sub_idx: int):
        """Make a ParallelReservoirsArguments object from a Config object.

        This method generates a ParallelReservoirsArguments object based on the provided Config object.
        The Config object should have keys that match the names defined in this class.

        Args:
            conf (Config): The Config object corresponding to the YAML for the parameter scan.
            idx (int): The index of the current parameter set.

        Returns:
            ParallelReservoirsArguments: The generated ParallelReservoirsArguments object.
        """
        parameters = conf.param_scan_list()[job_idx][sub_idx]

        if isinstance(parameters["identical_outputmatrix"], list):
            parameters["identical_outputmatrix"] = tuple(
                parameters["identical_outputmatrix"]
            )
            pass
        parameters["spatial_shape"] = tuple(parameters["spatial_shape"])
        parameters["parallelreservoirs_grid_shape"] = tuple(
            parameters["parallelreservoirs_grid_shape"]
        )
        del parameters["training_data_index"]
        return cls(**parameters)

    @classmethod
    def from_dict(cls, input_dict: dict):
        """Generate parameters from dictionary

        This function supports passing of a dictionary that contains more than the
        needed keys. In this case any additional information is simply ignored.

        Args:
            input_dict:
                dictionary containing at least all the needed keys for the
                initialiser.
        """
        class_fields = {f.name for f in fields(cls)}

        if isinstance(input_dict["identical_outputmatrix"], list):
            input_dict["identical_outputmatrix"] = tuple(
                input_dict["identical_outputmatrix"]
            )

        input_dict["spatial_shape"] = tuple(input_dict["spatial_shape"])

        if isinstance(input_dict["parallelreservoirs_grid_shape"], int):
            input_dict["parallelreservoirs_grid_shape"] = tuple(
                [input_dict["parallelreservoirs_grid_shape"]]
            )
        else:
            input_dict["parallelreservoirs_grid_shape"] = tuple(
                input_dict["parallelreservoirs_grid_shape"]
            )

        return cls(**{k: v for k, v in input_dict.items() if k in class_fields})


class ParallelReservoirsBase(ABC):
    """Base class for parallel reservoir applications. Implements the use of multiple Reservoir Computers in parallel to predict high dimensional (spatially extended) systems.
    Parallel is to be understood in terms of domain splitting of the input data."""

    def __init__(
        self,
        args: ParallelReservoirsArguments,
        **kwargs,
    ):
        """
        - Base class: Set up reservoir and parallel-reservoir parameters, initialize reservoirs and slices.

        Args:

            args (ParallelReservoirsArguments):
                All needed arguments in a dataclass object.
        """
        logger.log(loglevel_info_multiple_run, "Initializing ParallelReservoirsBase.")
        # set data input parameters
        self.spatial_shape = args.spatial_shape
        self.system_variables = args.system_variables
        self.boundary_condition = args.boundary_condition

        # set reservoir parameters
        self.adj_degree = args.adjacency_degree
        self.adj_dense = args.adjacency_dense
        self.adj_spectralrad = args.adjacency_spectralradius
        self.res_leakage = args.reservoir_leakage
        self.res_nodes = args.reservoir_nodes
        self.input_scaling = args.input_scaling
        self.input_bias = args.input_bias
        self.output_bias = args.training_output_bias

        # set multi-reservoir parameters
        self.rc_grid_shape = args.parallelreservoirs_grid_shape
        self.rc_ghosts = args.parallelreservoirs_ghosts

        # set dimensionality reduction parameters
        self.dr_fraction = args.dimensionreduction_fraction

        # set training parameters
        self.train_includeinput = args.training_includeinput
        self.train_regularization = args.training_regularization

        # how many matrices are used
        self.identical_input = args.identical_inputmatrix
        self.identical_adjacencymatrix = args.identical_adjacencymatrix
        self.identical_output = args.identical_outputmatrix

        # check if identical matrices are set correctly
        if isinstance(self.identical_output, (tuple, str)):
            if not self.identical_adjacencymatrix:
                self.identical_adjacencymatrix = True
                logger.warning(
                    "Identical output matrix requires identical adjacency matrix. Identical adjacency matrix is set to True."
                )
            if not self.identical_input:
                self.identical_input = True
                logger.warning(
                    "Identical output matrix requires identical input matrix. Identical input matrix is set to True."
                )

        # deprecated parameters (just in case)
        # self.dtype = args.dtype
        # self.deprecated_trainonly = args.deprecated_trainonly
        # self.train_overwrite = args.training_overwrite

        # Calculate the shape of the domain on which each reservoir predicts. To this shape the reservoirs output will be reshaped. Data of this shape is transformed and dimension reduced to create the input.
        self.res_domain_size = np.append(
            [self.system_variables],
            np.array(self.spatial_shape) // np.array(self.rc_grid_shape)
            + 2 * self.rc_ghosts,
        )
        # res_inpu_length depends on dimensionreduction and is set in the derived classes
        self.res_output_length = np.prod(self.res_domain_size)
        self.res_input_length = self._get_input_length()
        logger.log(
            loglevel_debug_parallelreservoirs,
            f"ParallelReservoir.__init__\t\t\tReservoir output shape: {self.res_domain_size}, Reservoir input length: {self.res_input_length}",
        )

        # TODO write seed somewhere!
        self.seed = np.random.randint(0, 2**32 - 1)
        self.reservoirs = self._initialize_reservoirs(seed=self.seed)

        # set up slices
        ## one prediction slice for all reservoirs
        if self.rc_ghosts == 0:
            self.remove_ghosts = tuple(
                slice(None) for _ in range(len(self.rc_grid_shape) + 2)
            )
        else:
            self.remove_ghosts = (slice(None), slice(None)) + tuple(
                slice(self.rc_ghosts, -self.rc_ghosts)
                for _ in range(len(self.rc_grid_shape))
            )
        ## one input slice for each reservoir: list of slicing objects
        self.reservoir_slices = self._initialize_reservoir_slices()
        logger.log(loglevel_info_multiple_run, "Done.")

    def train(
        self,
        input_training_data: np.ndarray,
        output_training_data: np.ndarray,
        transient_steps: int = 0,
    ) -> None:
        """Train the output matrix of all parallel reservoirs using ridge regression.
        Hereby it is differentiated between different methods:

        1. Each reservoir is trained individually. Works for arbirary systems.

        2. One reservoir (used for the predictions of all domains) is trained on one domain only. Works only for homogenious systems.

        3. One reservoir (used for the predictions of all domains) is trained on all domains by combigning the data. Works only for homogenious systems.

        Args:
            input_training_data: np.ndarray[float]
                input data for training, without bouncary condition ghostcells. Data needs to be of shape :code:`(time_steps, variables, *spatial_shape)`.
            output_training_data: np.ndarray[float]
                output data for training, without bouncary condition ghostcells. Data needs to be of shape :code:`(time_steps - transient_steps, variables, *spatial_shape)`.
            transient_steps: int, optional
                number of transient steps to be used for training (default: 0)

        Notes:
            The training method includes a transient phase, where the reservoirs are driven by the input data without using the results for the training.
        """
        logger.log(loglevel_info_multiple_run, f"Training parallel reservoirs.")

        logger.log(
            loglevel_debug_parallelreservoirs,
            f"ParallelReservoir.train\t\t\t\tTraining input, without ghostcells shape: {input_training_data.shape}",
        )
        logger.log(
            loglevel_debug_parallelreservoirs,
            f"ParallelReservoir.train\t\t\t\tTraining output, without ghostcells shape: {output_training_data.shape}",
        )
        input_training_data = self._boundary_condition(
            input_training_data,
            add_ghostcells=True,
        )
        output_training_data = self._boundary_condition(
            output_training_data,
            add_ghostcells=True,
        )

        logger.log(
            loglevel_debug_parallelreservoirs,
            f"ParallelReservoir.train\t\t\t\tTraining input, with ghost cells shape: {input_training_data.shape}",
        )
        logger.log(
            loglevel_debug_parallelreservoirs,
            f"ParallelReservoir.train\t\t\t\tTraining output, with ghosts cells shape: {output_training_data.shape}",
        )

        # Test data shapes
        if len(input_training_data.shape) != 2 + len(self.spatial_shape) or len(
            output_training_data.shape
        ) != 2 + len(self.spatial_shape):
            raise ValueError(
                f"Data must have shape (time_steps, variables, *spatial_shape), but have shape input_shape = {input_training_data.shape} and output_shape = {output_training_data.shape}."
            )
        if (
            input_training_data.shape[0]
            != output_training_data.shape[0] + transient_steps
        ):
            raise ValueError(
                f"Input and output training data must have the same length (only input is supposed to include transient). But input is of shape {input_training_data.shape}, output is of shape {output_training_data.shape} and transient_steps is {transient_steps}."
            )
        # Test training specifying variables:
        if isinstance(self.identical_output, tuple):
            if np.any(self.identical_output >= self.rc_grid_shape):
                raise ValueError(
                    f"Identical output reservoir index {self.identical_output} is out of bounds. Number of reservoirs is {np.prod(self.rc_grid_shape)}."
                )

        # distinguish between different trainings
        if self.identical_output is False:
            # train all reservoirs individually
            for i, res in enumerate(self.reservoirs):
                res.train(
                    input=self._transform_data(
                        input_training_data[self.reservoir_slices[i]],
                        fraction=self.dr_fraction,
                    ),
                    output=self._transform_data(
                        output_training_data[self.reservoir_slices[i]], fraction=1
                    ),
                    transient_steps=transient_steps,
                )
        elif isinstance(self.identical_output, tuple):
            # train one reservoir on one domain
            idx = np.ravel_multi_index(self.identical_output, self.rc_grid_shape)
            self.reservoirs[0].train(
                input=self._transform_data(
                    input_training_data[self.reservoir_slices[idx]],
                    fraction=self.dr_fraction,
                ),
                output=self._transform_data(
                    output_training_data[self.reservoir_slices[idx]], fraction=1
                ),
                transient_steps=transient_steps,
            )

        elif self.identical_output == "combine_data":
            # train one reservoir on all domains
            ## stack spatial data to a list of trainingdatasets
            input_data = [
                self._transform_data(
                    input_training_data[self.reservoir_slices[i]],
                    fraction=self.dr_fraction,
                )
                for i, _ in enumerate(self.reservoirs)
            ]
            output_data = [
                self._transform_data(
                    output_training_data[self.reservoir_slices[i]], fraction=1
                )
                for i, _ in enumerate(self.reservoirs)
            ]

            # train the first reservoir on all datasets
            self.reservoirs[0].train_on_multiple_datasets(
                inputs=input_data, outputs=output_data, transient_steps=transient_steps
            )
        else:
            raise ValueError(
                f"Identical output matrix needs to be False, tuple or 'combine_data', but is {self.identical_output}."
            )
        logger.log(loglevel_info_multiple_run, f"Done.")

    def iterative_predict(
        self, initial, max_steps, supervision_data=None, **kwargs
    ) -> tuple[np.ndarray, np.ndarray | None, int]:
        """Iteratively predict a time series.

        Args:
            initial (np.ndarray[float]):
                Initial condition for the time series prediction.
            max_steps (int):
                Maximum number of steps to predict.
            supervision_data (np.ndarray[float], optional):
                Supervision data for evaluating the prediction.
                Defaults to None.
            **kwargs:
                Additional keyword arguments for evaluating the prediction:
                Including an error function :code:`error_function` of {'NRMSE'},
                the (temporal) mean of the norm of the data :code:`mean_norm`,
                a threshhold value for the error function :code:`error_stop`,
                and the number of extra steps to predict after the error threshold is exceeded :code:`extra_steps`.

        Returns:
            tuple[np.ndarray, np.ndarray | None, int]:
                A tuple containing the predicted time series,
                the prediction errors (if supervision data is provided, else None),
                and the number of steps predicted.

        Notes: Before the prediction the parallel reservoir states need to be adjusted to the state of the system to be predicted by using :code:`reservoir_transient`.

        Warning: This function with supervision data is not tested nor fully implemented yet.
        """
        logger.log(loglevel_info_single_run, f"Iteratively predicting time series.")

        # add ghostcells to initial data to fullfill boundary condition, and create prediction place holder
        initial = self._boundary_condition(initial, add_ghostcells=True)
        prediction = np.empty((max_steps, *initial.shape[1:]))

        logger.log(
            loglevel_debug_parallelreservoirs,
            f"ParallelReservoir.iterative_predict\t\tInitial shape: {initial.shape}",
        )
        logger.log(
            loglevel_debug_parallelreservoirs,
            f"ParallelReservoir.iterative_predict\t\tPrediction shape: {prediction.shape}",
        )

        # Decide whether to predict with supervision data or not, if not max_steps time steps are predicted, else the _evaluate_one_step function is used to stop the prediction
        if supervision_data is None:
            for t in range(max_steps):
                for i, res in enumerate(self.reservoirs):
                    prediction[t : t + 1][self.reservoir_slices[i]][
                        self.remove_ghosts
                    ] = self._inv_transform_data(
                        res.predict_single_step(
                            self._transform_data(
                                initial[self.reservoir_slices[i]],
                                fraction=self.dr_fraction,
                            )
                        )
                    )[
                        self.remove_ghosts
                    ]

                initial = self._boundary_condition(prediction[t : t + 1])

            # return inner prediction/ neglect ghostcells
            logger.log(loglevel_info_single_run, f"Done.")
            return (prediction[self.remove_ghosts], None, max_steps)

        else:
            # TODO: This could be modified. I test here all arguments needed for evaluation of a prediction.
            # TODO: The parameters are set as kwargs, because they dont belong to the class, but to this prediciton

            # Checking arguments for iterative predction
            try:
                error_function = kwargs["error_function"]
                error_stop = kwargs["error_stop"]
            except KeyError:
                raise ValueError(
                    "Errorfunction 'error_function' and threshhold value 'error_stop' need to be set to evaluate predictions."
                )
            # setting number of steps after threshold is exceeded
            if "extra_steps" in kwargs:
                extra_steps = kwargs["extra_steps"]
            else:
                extra_steps = 0  # or any default value

            # setting error function
            if error_function == "NRMSE":
                try:
                    mean_norm = kwargs["mean_norm"]

                except KeyError:
                    raise ValueError(
                        "Mean norm of data need to be set as keyword arguments to evaluate predictions."
                    )
            else:
                raise NotImplementedError(
                    "Only Normalized Root Mean Square Error 'NRMSE' is implemented so far."
                )

            # create variables
            errors = np.zeros(max_steps)
            good_prediction = True

            # iteratively predict and measure error
            t = 0
            while t < max_steps and good_prediction:
                for i, res in enumerate(self.reservoirs):
                    prediction[t : t + 1][self.reservoir_slices[i]][
                        self.remove_ghosts
                    ] = self._inv_transform_data(
                        res.predict_single_step(
                            self._transform_data(
                                initial[self.reservoir_slices[i]],
                                fraction=self.dr_fraction,
                            )
                        )
                    )[
                        self.remove_ghosts
                    ]

                initial = self._boundary_condition(prediction[t : t + 1])

                # measure error, prediction slice is used to compare only the inner part of the data/ neglect ghost cells
                good_prediction, errors[t] = self._evaluate_one_step(
                    prediction[t : t + 1][self.remove_ghosts],
                    supervision_data[t],
                    errrofunction=error_function,
                    mean_norm=mean_norm,
                    error_stop=error_stop,
                )
                # TODO: Hier muessen noch parameter von _evaluate_one_step gesetzt werden, die nicht allgemein in der baseclass sein muessen: Kwargs?
                # TODO: Es gibt hier auch noch das problem, dass eigentlich nur auf dem inneren, i.e. ohne ghostcells der fehler evaluiert werden solte
                t += 1
            t2 = t
            while t2 < max_steps and t2 < t + extra_steps:
                for i, res in enumerate(self.reservoirs):
                    prediction[t2 : t2 + 1][self.reservoir_slices[i]][
                        self.remove_ghosts
                    ] = self._inv_transform_data(
                        res.predict_single_step(
                            self._transform_data(
                                initial[self.reservoir_slices[i]],
                                fraction=self.dr_fraction,
                            )
                        )
                    )[
                        self.remove_ghosts
                    ]

                initial = self._boundary_condition(prediction[t2 : t2 + 1])
                # measure error, prediction slice is used to compare only the inner part of the data/ neglect ghost cells
                _, errors[t2] = self._evaluate_one_step(
                    prediction[t2 : t2 + 1][self.remove_ghosts],
                    supervision_data[t2 : t2 + 1],
                    errrofunction=error_function,
                    mean_norm=mean_norm,
                    error_stop=error_stop,
                )
                # TODO: Hier muessen noch parameter von _evaluate_one_step gesetzt werden, die nicht allgemein in der baseclass sein muessen: Kwargs?
                # TODO: Es gibt hier auch noch das problem, dass eigentlich nur auf dem inneren, i.e. ohne ghostcells der fehler evaluiert werden solte
                t2 += 1
            logger.log(loglevel_info_single_run, f"Done.")
            return (prediction[self.remove_ghosts][:t2], errors[:t2], t)

    def reservoir_transient(
        self, input, predict_on_transient=False
    ) -> None | np.ndarray:
        """Transient dynamics of the reservoirs. Updates the reservoir states and optionally predicts one step ahead on the transient dynamics, without reusing the predictions.

        Args:
            input: np.ndarray[float]
                input data for the transient dynamics
            predict_on_transient: bool
                if True, one step-ahead predictions are performed on transient data and returned, else only the reservoir states are updated
        """

        logger.log(
            loglevel_info_single_run, f"Updating reservoir nodes on transient dynamics."
        )

        logger.log(
            loglevel_debug_parallelreservoirs,
            f"ParallelReservoir.reservoir_transient\t\tInput shape: {input.shape}",
        )
        # add ghostcells to input data to fullfill boundary condition
        input = self._boundary_condition(
            input,
            add_ghostcells=True,
        )

        # destinguish, whether to predict on transient or not
        if predict_on_transient:
            prediction = np.empty(input.shape)
            logger.log(
                loglevel_debug_parallelreservoirs,
                f"ParallelReservoir.reservoir_transient\t\tPrediction shape: {prediction.shape}",
            )

            # predict one step ahead on transient data, without reusing the predictions
            ## the order of the loops is irelevant, because the reservoirs are not interacting, but all time steps and all reservoirs need to be used
            for t in range(input.shape[0]):
                for i, res in enumerate(self.reservoirs):
                    prediction[t : t + 1][self.reservoir_slices[i]][
                        self.remove_ghosts
                    ] = self._inv_transform_data(
                        res.predict_single_step(
                            self._transform_data(
                                input[t : t + 1][self.reservoir_slices[i]],
                                fraction=self.dr_fraction,
                            )
                        )
                    )[
                        self.remove_ghosts
                    ]
            logger.log(loglevel_info_single_run, f"Done.")
            return prediction[
                self.remove_ghosts
            ]  # return prediction without ghostcells
        else:
            # Update the reservoir states on the transient data
            # All time steps can be used at ones, because no interaction of the reservoirs is used
            for i, res in enumerate(self.reservoirs):
                for input_t in self._transform_data(
                    input[self.reservoir_slices[i]], fraction=self.dr_fraction
                ):
                    res.propagate_reservoir_state(input_t)
            logger.log(loglevel_info_single_run, f"Done.")
            return None

    def _initialize_reservoirs(self, seed: int) -> list[ReservoirComputer]:
        """Initialize the reservoirs for the parallel reservoirs using :code:`rc_grid_shape`, which specifies how many reservoirs in each dimension are used.

        Args:
            seed (int): Seed for the random number generator, used for initializing the reservoirs.

        Returns:
            list[ReservoirComputer]: The list of initialized reservoirs.
        """

        np.random.seed(seed=seed)
        return [
            ReservoirComputer(
                nodes=self.res_nodes,
                degree=self.adj_degree,
                leakage=self.res_leakage,
                spectral_radius=self.adj_spectralrad,
                input_scaling=self.input_scaling,
                input_length=self.res_input_length,
                input_bias=self.input_bias,
                output_bias=self.output_bias,
                regularization=self.train_regularization,
                training_includeinput=self.train_includeinput,
                identical_inputmatrix=self.identical_input,
                identical_adjacencymatrix=self.identical_adjacencymatrix,
                identical_outputmatrix=self.identical_output,
                reset_matrices=(i == 0),
            )
            for i in range(np.prod(self.rc_grid_shape))
        ]

    def _initialize_reservoir_slices(self) -> list[tuple[slice]]:
        r"""Prepare the reservoir slices for different reserovirs. First two dimensions are ignored, because they are :code:`(time_steps, variables)`/ not spatial.
        In each spatial dimension of length :math:`dim`, the slices for reservoir number :math:`i` (w.r.t. this dimension) of total number of reservoirs :math:`n` is chosen to be :code:`slice(start : stop)`, where

        .. math::

            \text{start} = i \cdot \left\lfloor\frac{dim}{n}\right\rfloor\quad\text{and}\quad\text{stop} = (i+1) \cdot \left\lfloor\frac{dim}{n}\right\rfloor + 2 \cdot \text{ghosts}.

        Returns:
            list[tuple[slice]]: List of slices for each reservoir
        """
        # TODO change name 'data_slices?'
        # NOTE This only cares about spatial dimensions so far. Temporal and variable dimensions should be added here or need to be added in the application?
        slices = []

        # loop over all reservoirs
        for i, res in enumerate(self.reservoirs):
            mult_index = np.unravel_index(i, self.rc_grid_shape)

            # create slice
            slices.append(
                (slice(None), slice(self.system_variables))
                + tuple(
                    slice(
                        mult_index[dim]
                        * (self.spatial_shape[dim] // self.rc_grid_shape[dim]),
                        (mult_index[dim] + 1)
                        * (self.spatial_shape[dim] // self.rc_grid_shape[dim])
                        + 2 * self.rc_ghosts,
                    )
                    for dim in range(len(self.spatial_shape))
                )
            )
        return slices

    # TODO: JIT
    def _boundary_condition(
        self,
        data: np.ndarray,
        add_ghostcells: bool = False,
    ) -> np.ndarray:
        """The function enforces boundary conditions on all spatial dimensions and returns the result.
            Either boundary cells are used as ghost cells or ghost cells are added.
            Ghost cell values depend on the boundary condition.

        Args:
            data (np.ndarray): Input data in region to be predicted. Shape is :code:`(time_steps, variables, *spatial_shape)`.
            add_ghostcells (bool): If True, the boundary condition is fullfilled by adding ghostcells, else the outercells of the array are updated to fullfill the boundary condition.

        Return:
            np.ndarray: The extended array of size :code:`data.shape` or :code:`data.shape+2*window_size`, depending on :code:`add_ghostcells`.

        Notes:
            The boundary condition is applied to spatial dimensions only. Therefore, the first and second dimension of the input :code:`data` is ignored.

        Attention:
            This has not been tested for arbitrary dimensions. Only up to 2D.
            Might be useful to be precompiled(Numba-JIT), it runs in every time step
        """

        shape = np.array(data.shape)
        if add_ghostcells:
            shape[2:] += 2 * self.rc_ghosts
            increased_data = np.empty(shape)

            # interior
            # ind = [slice(self.rc_ghosts, -self.rc_ghosts) for _ in range(len(shape))]
            # ind[0], ind[1] = slice(shape[0]), slice(shape[1])
            increased_data[self.remove_ghosts] = data
        else:
            increased_data = data
        # boundary
        if self.boundary_condition == "Periodic":
            dim = 0
            while dim < len(shape) - 2:
                increased_data[:, :, : self.rc_ghosts] = increased_data[
                    :, :, -2 * self.rc_ghosts : -self.rc_ghosts
                ]
                increased_data[
                    :, :, increased_data.shape[2] - self.rc_ghosts :
                ] = increased_data[:, :, self.rc_ghosts : 2 * self.rc_ghosts]
                increased_data = np.moveaxis(increased_data, 2, -1)
                dim += 1
        elif self.boundary_condition == "NoFlux":
            dim = 0
            while dim < len(shape) - 2:
                increased_data[:, :, : self.rc_ghosts] = increased_data[
                    :, :, 2 * self.rc_ghosts - 1 : self.rc_ghosts - 1 : -1
                ]
                increased_data[
                    :, :, increased_data.shape[2] - self.rc_ghosts :
                ] = increased_data[
                    :, :, -self.rc_ghosts - 1 : -2 * self.rc_ghosts - 1 : -1
                ]
                increased_data = np.moveaxis(increased_data, 2, -1)
                dim += 1
        return increased_data

    # TODO: JIT
    @staticmethod
    def _evaluate_one_step(
        prediction: np.ndarray,
        supervision_data: np.ndarray,
        errrofunction="NRMSE",
        mean_norm: float = 1,
        error_stop: float = 1,
    ) -> tuple[bool, float]:
        r"""Evaluate the prediction of one time step using the error function :code:`errorfunction`.
        If :code:`errorfunction ="NRMSE"`, the normalized root mean square error

        .. math::

            \frac{\|\vec{u}(t)-\vec{u}^{\mathrm{true}}(t)\|_2}{\langle\|\vec{u}^{\mathrm{true}}(t)\|^2\rangle_{\mathrm{t}}^{1/2}}

        is used.

        Args:
            prediction (np.ndarray):
                Prediction of the system at one time step. Shape is :code:`(variables, *spatial_shape)`.
            supervision_data (np.ndarray):
                Supervision data for the prediction. Shape is :code:`(variables, *spatial_shape)`.
            errorfunction (str):
                Error function to evaluate the prediction. Only the nomalized root mean square error :code:`"NRMSE"` is implemented so far.
            mean_norm (float):
                Mean norm of the supervision data, is used to normalize the error root mean square error.
            error_stop (float):
                Threshold for the error function. Iterative predictions are stopped if the error is above the threshold.

        Returns:
            bool:
                True if the error is below the threshold :code:`error_stop`, else False.

        Warning: This method is not tested and might have errors!
        """
        # print(prediction.shape, supervision_data.shape)
        if errrofunction == "NRMSE":
            error = float(np.linalg.norm(prediction - supervision_data) / mean_norm)
            return bool(error_stop > error), error
        else:
            raise NotImplementedError(
                "Only Normalized Root Mean Square Error 'NRMSE' is implemented so far."
            )

    @abstractmethod
    def _get_input_length(self) -> int:
        """Depends on dimension reduction method.

        Get the length of the input data for each parallel object of class :code:`ReservoirComputer`. Depends on Dimension Reduction fraction.

        Returns:
            int: The length of the input data for each :code:`ReservoirComputer`
        """
        pass

    @abstractmethod
    def _transform_data(self, data: np.ndarray, fraction: float) -> np.ndarray:
        """Depends on dimension reduction method.

        Transform the data to the shape used in the reservoirs.
        Data of shape :code:`(time_steps, variables, *spatial_shape)` is transformed to :code:`(time, res_variables)`, where all variables and spatial shapes are used in the dimension reduction and flatted into one dimension.

        Args:
            data (np.ndarray):
                Data to be transformed. First dimension is temporal and not transformed. Second dimension chooses different variables and will be used in dimension reduction. All others are spatial dimensions and are flatted as well.
            fraction (float):
                Fraction of variables that actually enters the reservoir after dimension reduction.
        Returns:
            np.ndarray: Transformed data. First dimension is temporal and not transformed. Rest is flattened.
        """
        pass

    @abstractmethod
    def _inv_transform_data(self, data: np.ndarray) -> np.ndarray:
        """Depends on dimension reduction method.

        Inverse dimension reduction transformation of the flatted reservoir data :code:`data`.
        The output is transformed to the geometric shape of the output prediction.
        First dimension of input :code:`data` is temporal and not transformed. Second dimension is split into different variables and spatial dimensions.

        Args:
            data (np.ndarray):
                Data to be transformed. First dimension is temporal and not transformed.

        Returns:
            np.ndarray: Inverse transformed data. First dimension is temporal and not transformed. Rest is reshaped to number of variables and spatial shape.

        """
        pass

    def save(self, filename: str) -> None:
        """Save a ParallelReservoir to pkl file

        A possible use case of this is to conserve a trained multi-reservoir for later.

        Args:
            filename (str):
                name of file to generate

        Warning: This function might be depracated. Else, might need to be put to derived classes.
        """
        pickle.dump(self, open(filename, "wb"))

    def load(self, filename: str) -> None:
        """Load a ParallelReservoir from pkl file

        A possible use case of this is to load a trained model and perform further
        predictions without having to do the training again.

        Args:
            filename (str):
                name of file to read in

        Warning: This function might be depracated. Else, might need to be put to derived classes.
        """
        self = pickle.load(open(filename, "rb"))


class ParallelReservoirs(ParallelReservoirsBase):
    """Use multiple Reservoir Computers in parallel to predict high dimensional systems without dimensionality reduction.
    Parallel is to be understood in terms of domain splitting of the input data.
    This class uses the base class :code:`ParallelReservoirsBase`, which initiates Reservoirs of the class :code:`ReservoirComputer` and handles the training and prediction of the parallel reservoirs.

    **Initialization**:

    - Only base class setup is done. No further initialization is needed.
    """

    def __init__(
        self,
        *,
        Parameter: ParallelReservoirsArguments,
        **kwargs,
    ):
        logger.log(loglevel_info_multiple_run, f"Setting up parallel reservoirs.")
        # init base class
        super().__init__(Parameter, **kwargs)
        if self.dr_fraction != 1:
            raise ValueError(
                "Class ParallelReservoirs should not have dimension reduction, but got dr_fraction != 1."
            )
        logger.log(loglevel_info_multiple_run, f"Done.")

    def _get_input_length(self) -> int:
        r"""Get the length of the input data for each parallel object of class :code:`ReservoirComputer`. Without dimension reduction, the input length is

        .. math::

            \text{input length} = \prod_{i=1}^{d} \left( \frac{N_i}{R_i} + 2 g\right)

        where :math:`d` is the number of spatial dimensions of the predicted system, :math:`N_i` is the number of support points in the :math:`i`-th dimension, :math:`R_i` is the number of parallel reservoirs in the :math:`i`-th dimension, :math:`g` is the number of ghost cells used aroud the predicted region of each reservoir.

        Returns:
            int: The length of the input data for each :code:`ReservoirComputer`.
        """
        return np.prod(self.res_domain_size, dtype=int)

    def _transform_data(self, data: np.ndarray, fraction: float) -> np.ndarray:
        """Transform the data to the shape used in the reservoirs.

        Args:
            data (np.ndarray):
                Data to be transformed. First dimension is temporal and not transformed. Second dimension chooses different variables and will be flattend. All others are spatial dimensions and are flatted as well.
            fraction (float):
                Fraction of variables that actually enters the reservoir after dimension reduction. Should be one. Is not used.
        Returns:
            np.ndarray: Transformed data. First dimension is temporal and not transformed. Rest is flattened.
        """
        logger.log(
            loglevel_debug_DRRC,
            f"ParallelReservoir._transform_dataTransormation\tInput shape: {data.shape} and transformed shape {data.reshape((data.shape[0], -1)).shape}",
        )
        return np.reshape(data, (data.shape[0], -1))

    def _inv_transform_data(self, data) -> np.ndarray:
        """Inverse transformation of the data flatted reservoir data to the geometric shape. First dimension of input is temporal and not transformed. Second dimension is split into different variables and spatial dimensions.

        Args:
            data (np.ndarray):
                Data to be transformed. First dimension is temporal and not transformed.

        Returns:
            np.ndarray: Transformed data. First dimension is temporal and not transformed. Rest is reshaped to number of variables and spatial shape.

        """
        logger.log(
            loglevel_debug_DRRC,
            f"ParallelReservoir._inv_transform_data\t\tInverse transormation input shape: {data.shape} and transformed shape {data.reshape((data.shape[0], -1, *self.res_domain_size)).shape}",
        )
        return np.reshape(data, (data.shape[0], *self.res_domain_size))


class ParallelReservoirsFFT(ParallelReservoirsBase):
    """Use multiple Reservoir Computers in parallel to predict high dimensional systems using largest modes of an FFT for dimensionality reduction.
    Parallel is to be understood in terms of domain splitting of the input data.
    This class uses the base class :code:`ParallelReservoirsBase`, which initiates Reservoirs of the class :code:`ReservoirComputer` and handles the training and prediction of the parallel reservoirs.
    """

    def __init__(
        self,
        *,
        Parameter: ParallelReservoirsArguments,
        prediction_model: str,
        **kwargs,
    ):
        """
        **Initialization**:

        - Calculates or loads the largest FFT modes for the given datatype, i.e. model and parameters, and input and output dimension for each parallel Reservoir Computer.

        - Base class: Set up reservoir and parallel-reservoir parameters, initialize reservoirs and slices.

        Args:

            args (ParallelReservoirsArguments):
                All needed arguments in a dataclass object.
            prediction_model (str):
                Name of the model to be predicted. Used to load or choose FFT modes with largest amplitude.
            **kwargs : dict
                nr_fft_datasets (int):
                    The number of training data sets used to train the FFT. If not used all 10 training data sets are used.
                    This is useful for computations with not enough memory for the full training.

        Warning:
            The same modes are used everywhere and choosen on all training data sets and all domains.
            Hence, the class only works if two conditions are met:
            1. The predicted Systems are homogeneous.
            2. All data sets need to be sampled from the same attractor.
        """
        super().__init__(Parameter, **kwargs)

        # precompute tuple of spatial axes
        self.spatial_axes = tuple(2 + np.arange(len(self.spatial_shape)))

        # init list of indices of largest modes (will be overwritten with ordered list, decreasing)
        ## The initialization here is needed to run the transform function, to calculate the modes
        self.largest_modes = tuple(range(self.res_output_length))

        filename = prediction_model + "_" + str(self.res_domain_size)
        fft_path = str(Config.get_git_root()) + "/Data/FFT/"

        if "nr_fft_datasets" in kwargs:
            if kwargs["nr_fft_datasets"] > 10:
                raise ValueError(
                    "Number of training data sets for PCA can be 10 at max."
                )
            nr_trainings = kwargs["nr_fft_datasets"]
            filename += f"_trainedon{nr_trainings}"
        else:
            nr_trainings = 10
        temp_skip = 2

        if paths.isfile(fft_path + filename + ".pkl"):
            logger.log(
                loglevel_info_multiple_run,
                f"Loading FFT modes for {prediction_model} with domain size {self.res_domain_size} from {fft_path+filename}.",
            )
            self.largest_modes = pickle.load(open(fft_path + filename + ".pkl", "rb"))
        else:
            logger.log(
                loglevel_info_multiple_run,
                f"Choosing FFT modes for {prediction_model} with domain size {self.res_domain_size}.",
            )
            if prediction_model == "1D_KuramotoSivashinsky":
                max_len = 80000 // (len(self.reservoirs))
                logger.log(
                    loglevel_info_multiple_run,
                    f"Gather triningdata.",
                )
                fft_data = np.concatenate(
                    [
                        self._boundary_condition(
                            data=np.load(
                                str(Config.get_git_root())
                                + f"/Data/{prediction_model}/TrainingData{i}.npy"
                            )[:max_len:temp_skip, np.newaxis, :],
                            add_ghostcells=True,
                        )
                        for i in range(nr_trainings)
                    ],
                    axis=0,
                )
                fft_data = np.concatenate(
                    [
                        fft_data[self.reservoir_slices[i]]
                        for i in range(len(self.reservoir_slices))
                    ],
                    axis=0,
                )
            elif prediction_model == "2D_AlievPanfilov":
                max_len = 20000 // (len(self.reservoirs))
                logger.log(
                    loglevel_info_multiple_run,
                    f"Gather triningdata.",
                )
                fft_data = np.concatenate(
                    [
                        self._boundary_condition(
                            data=np.load(
                                str(Config.get_git_root())
                                + f"/Data/{prediction_model}/TrainingData{i}.npz"
                            )["vars"][:max_len:temp_skip, : self.system_variables],
                            add_ghostcells=True,
                        )
                        for i in range(nr_trainings)
                    ],
                    axis=0,
                )
                fft_data = np.concatenate(
                    [
                        fft_data[self.reservoir_slices[i]]
                        for i in range(len(self.reservoir_slices))
                    ],
                    axis=0,
                )
            else:
                raise FileNotFoundError(
                    f"Training data for {prediction_model} not found."
                )

            logger.log(
                loglevel_info_multiple_run,
                f"Choosing FFT modes for domain size {self.res_domain_size} with {fft_data.shape[0]} Samples/ timesteps.",
            )

            # select temporal max of largest modes
            self.largest_modes = np.argsort(
                np.amax(self._transform_data(fft_data, 1), axis=0)
            )[::-1]
            logger.log(
                loglevel_info_multiple_run,
                f"Saving FFT modes for {prediction_model} with domain size {self.res_domain_size} at {fft_path+filename}.",
            )
            pickle.dump(self.largest_modes, open(fft_path + filename + ".pkl", "wb"))

        # create object to invert the ordering of the largest modes
        self.invert_largest_modes = np.arange(len(self.largest_modes))
        np.put(
            self.invert_largest_modes,
            self.largest_modes,
            np.arange(self.largest_modes.size),
        )

        self.inv_largest_modes = np.argsort(self.largest_modes)
        # save an ordered list of indices of the largest modes
        # take only self.dimensionreduction_fraction of this ordered list --> cookie cutter for transform

    def _get_input_length(self) -> int:
        r"""Get the length of the input data for each parallel object of class :code:`ReservoirComputer`. With dimension reduction, the input length is

        .. math::

            \text{input length} = \left\lfloor f \times \prod_{i=1}^{d} \left( \frac{N_i}{R_i} + 2g \right) \right\rfloor

        where :math:`f` is the fraction :code:`self.dimensionreduction_fraction` of the dimensions used, :math:`d` is the number of spatial dimensions of the predicted system, :math:`N_i` is the number of support points in the :math:`i`-th dimension, :math:`R_i` is the number of parallel reservoirs in the :math:`i`-th dimension, :math:`g` is the number of ghost cells used aroud the predicted region of each reservoir.

        Returns:
            int: The length of the input data for each :code:`ReservoirComputer`.
        """
        return int(self.dr_fraction * self.res_output_length)

    def _transform_data(self, data: np.ndarray, fraction: float) -> np.ndarray:
        """Transform and reduce the data using the largest modes of the FFT. Flattening all but the first dimension to the shape used in the reservoirs.

        Args:
            data (np.ndarray):
                Data to be transformed. First dimension is temporal and not transformed. Second dimension chooses different variables, all others are spatial dimensions.
            fraction (float):
                Fraction of variables that actually enters the reservoir after dimension reduction.
        Returns:
            np.ndarray: Transformed data. First dimension is temporal and not transformed. Rest is flattened.

        Warning: Tested only in 1d
        """

        res_data = np.fft.rfftn(data, axes=self.spatial_axes)

        # Create tests if the following arrays are zeros everywhere, for different shapes. Else we discard this information
        # print(res_data.imag[..., 0])
        # print(res_data.imag[..., np.ceil(self.res_domain_size[-1]/2).astype(int):])

        # last axis is always factor 2 shorter do to symmetry of real data
        return np.concatenate(
            [
                res_data.real,
                res_data.imag[
                    ..., 1 : np.ceil(self.res_domain_size[-1] / 2).astype(int)
                ],
            ],
            axis=-1,
        ).reshape(data.shape[0], -1)[
            :, self.largest_modes[: int(self.res_output_length * fraction)]
        ]  # only take the first fraction of the largest modes

    def _inv_transform_data(self, data: np.ndarray) -> np.ndarray:
        """Inverse FFT transformation of the data flatted reservoir output (all components need to be predicted and are realigned to spatial shape). Transformed back to the geometric shape.
        This includes the boundary which will not be used when stitching the reservoir predictions together.
        First dimension of input is temporal and not transformed. Second dimension is split into different variables and spatial dimensions.

        Args:
            data (np.ndarray):
                Data to be transformed, of shape :code:`(time_steps, reservoir_variables)`. First dimension is temporal and not transformed.

        Returns:
            np.ndarray: Transformed data of shape :code:`(time_steps, variables, *spatial_shape)`. First dimension is temporal and not transformed. Rest is reshaped to number of variables and spatial shape.

        Notes:
            The input data is the output of one reservoir, which corresponds to one spatial domain.

        Warning: Tested only in 1d.
        """

        # undo ordering of largest modes
        data = data[..., self.inv_largest_modes]

        # reshape to spatial shape
        ## first half of last axes contains real part, second half imaginary part
        data = data.reshape(data.shape[0], *self.res_domain_size)

        # create complex data, last axis is always factor 2 shorter do to symmetry of real data
        fft_components = int(self.res_domain_size[-1] // 2 + 1)
        cmplx_data = np.zeros((*data.shape[:-1], fft_components), dtype=np.complex64)
        cmplx_data.real = data[..., :fft_components]
        cmplx_data.imag[
            ..., 1 : np.ceil(self.res_domain_size[-1] / 2).astype(int)
        ] = data[..., fft_components:]

        # inverse fft
        return np.fft.irfftn(cmplx_data, axes=self.spatial_axes).reshape(
            data.shape[0], *self.res_domain_size
        )


class ParallelReservoirsPCA(ParallelReservoirsBase):
    """Use multiple Reservoir Computers in parallel to predict high dimensional systems with using a PCA for dimensionality reduction.
    Parallel is to be understood in terms of domain splitting of the input data.
    This class uses the base class :code:`ParallelReservoirsBase`, which initiates Reservoirs of the class :code:`ReservoirComputer` and handles the training and prediction of the parallel reservoirs.

    """

    def __init__(
        self,
        *,
        Parameter: ParallelReservoirsArguments,
        prediction_model: str,
        **kwargs,
    ):
        r"""
        **Initialization**:

        - Trains or loads the PCA object for the given datatype, i.e. model and parameters, and output dimension for each parallel Reservoir Computer.
        Due to memory constrains, the pca-training data is taken from 10 training data sets and only each second time step is used. In addition for multiple parallel reservoirs the training data is sampled from all domains. In this case, the number of timesteps per domain is shortened (devided by number of domains) to keep the number of timesteps independet of the number of parallel reservoirs.
        Computational Complexity: :math:`\mathcal{O}(n\cdot d^2)`, where :math:`n` is the number of samples and :math:`d` is the number of dimensions.

        - Base class: Set up reservoir and parallel-reservoir parameters, initialize reservoirs and slices.

        Args:
            args (ParallelReservoirsArguments):
                All needed arguments in a dataclass object.
            prediction_model (str):
                Name of the model to be predicted. Used to load or train the PCA object.
            **kwargs : dict
                nr_pca_trainings (int):
                    The number of training data sets used to train the PCA. If not used all 10 training data sets are used.
                    This is useful for computations with not enough memory for the full training.

        Warning:
            The same PCA is used everywhere and trained on all training data sets and all domains.
            Hence, the class only works if two conditions are met:
            1. The predicted Systems are homogeneous.
            2. All data sets need to be sampled from the same attractor.

        Warining:
            Training PCAs with many parallel reservoirs and large ghost cells can be memory intensive.
        """
        # init base class
        super().__init__(args=Parameter)

        filename = prediction_model + "_" + str(self.res_domain_size)
        pca_path = str(Config.get_git_root()) + "/Data/PCA/"

        if "nr_pca_trainings" in kwargs:
            if kwargs["nr_pca_trainings"] > 10:
                raise ValueError(
                    "Number of training data sets for PCA can be 10 at max."
                )
            nr_trainings = kwargs["nr_pca_trainings"]
            filename += f"_trainedon{nr_trainings}"
        else:
            nr_trainings = 10
        temp_skip = 2

        if paths.isfile(pca_path + filename + ".pkl"):
            logger.log(
                loglevel_info_multiple_run,
                f"Loading PCA for {prediction_model} with domain size {self.res_domain_size} from {pca_path+filename}.",
            )
            self.pca = pickle.load(open(pca_path + filename + ".pkl", "rb"))
        else:
            logger.log(
                loglevel_info_multiple_run,
                f"Training PCA for {prediction_model} with domain size {self.res_domain_size}.",
            )
            if prediction_model == "1D_KuramotoSivashinsky":
                max_len = 80000 // len(self.reservoirs)
                logger.log(
                    loglevel_info_multiple_run,
                    f"Gather pca triningdata.",
                )
                pca_trainingdata = np.concatenate(
                    [
                        self._boundary_condition(
                            data=np.load(
                                str(Config.get_git_root())
                                + f"/Data/{prediction_model}/TrainingData{i}.npy"
                            )[:max_len:temp_skip, np.newaxis, :],
                            add_ghostcells=True,
                        )
                        for i in range(nr_trainings)
                    ],
                    axis=0,
                )
                pca_trainingdata = np.concatenate(
                    [
                        pca_trainingdata[self.reservoir_slices[i]]
                        for i in range(len(self.reservoir_slices))
                    ],
                    axis=0,
                )

            elif prediction_model == "2D_AlievPanfilov":
                max_len = 20000 // (len(self.reservoirs))
                logger.log(
                    loglevel_info_multiple_run,
                    f"Gather pca triningdata.",
                )
                pca_trainingdata = np.concatenate(
                    [
                        self._boundary_condition(
                            data=np.load(
                                str(Config.get_git_root())
                                + f"/Data/{prediction_model}/TrainingData{i}.npz"
                            )["vars"][:max_len:temp_skip, : self.system_variables],
                            add_ghostcells=True,
                        )
                        for i in range(nr_trainings)
                    ],
                    axis=0,
                )
                pca_trainingdata = np.concatenate(
                    [
                        pca_trainingdata[self.reservoir_slices[i]]
                        for i in range(len(self.reservoir_slices))
                    ],
                    axis=0,
                )
            else:
                raise FileNotFoundError(
                    f"Training data for {prediction_model} not found."
                )

            logger.log(
                loglevel_info_multiple_run,
                f"Training PCA for domain size {self.res_domain_size} with {pca_trainingdata.shape[0]} Samples.",
            )
            self.pca = PCA()
            self.pca.fit(pca_trainingdata.reshape(pca_trainingdata.shape[0], -1))
            logger.log(
                loglevel_info_multiple_run,
                f"Saving PCA for {prediction_model} with domain size {self.res_domain_size} at {pca_path+filename+'.pkl'}.",
            )
            pickle.dump(self.pca, open(pca_path + filename + ".pkl", "wb"))

    def _get_input_length(self) -> int:
        r"""Get the length of the input data for each parallel object of class :code:`ReservoirComputer`. With dimension reduction, the input length is

        .. math::

            \text{input length} = \left\lfloor f \times \prod_{i=1}^{d} \left( \frac{N_i}{R_i} + 2g \right) \right\rfloor

        where :math:`f` is the fraction :code:`self.dimensionreduction_fraction` of the dimensions used, :math:`d` is the number of spatial dimensions of the predicted system, :math:`N_i` is the number of support points in the :math:`i`-th dimension, :math:`R_i` is the number of parallel reservoirs in the :math:`i`-th dimension, :math:`g` is the number of ghost cells used aroud the predicted region of each reservoir.

        Returns:
            int: The length of the input data for each :code:`ReservoirComputer`.
        """
        return int(self.dr_fraction * self.res_output_length)

    def _transform_data(self, data: np.ndarray, fraction: float) -> np.ndarray:
        """Transform and reduce the data using the largest explained variances of the PCA. Flattening all but the first dimension to the shape used in the reservoirs.

        Args:
            data (np.ndarray):
                Data to be transformed. First dimension is temporal and not transformed. Second dimension chooses different variables, all others are spatial dimensions.
            fraction (float):
                Fraction of variables that actually enters the reservoir after dimension reduction.
        Returns:
            np.ndarray:
                Transformed data. First dimension is temporal and not transformed. Rest is flattened.

        Warning: Not implemented
        """

        return self.pca.transform(data.reshape(data.shape[0], -1))[
            :, : int(fraction * self.res_output_length)
        ]

    def _inv_transform_data(self, data: np.ndarray) -> np.ndarray:
        """Inverse PCA transformation of the data flatted reservoir output (all components need to be predicted). Transformed back to the geometric shape.
        This includes the boundary which will not be used when stitching the reservoir predictions together.
        First dimension of input is temporal and not transformed. Second dimension is split into different variables and spatial dimensions.

        Args:
            data (np.ndarray):
                Data to be transformed, of shape :code:`(time_steps, reservoir_variables)`. First dimension is temporal and not transformed.

        Returns:
            np.ndarray: Transformed data of shape :code:`(time_steps, variables, *spatial_shape)`. First dimension is temporal and not transformed. Rest is reshaped to number of variables and spatial shape.

        Notes:
            The input data is the output of one reservoir, which corresponds to one spatial domain.
        """

        return np.reshape(
            self.pca.inverse_transform(data), (data.shape[0], *self.res_domain_size)
        )
