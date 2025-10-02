# import logging as log
import logging as log
from warnings import warn

import numpy as np
import scipy
import scipy.sparse
from scipy.sparse.linalg import eigs
from sklearn.linear_model import Ridge

from drrc.tools.logger_config import drrc_logger as logger

# NOTE: Generally the reservoir computer gets data of the shape (time, dims), i.e. len(shape)==2 always is true
# NOTE: hstack should be removed from the predict_single_step function, as it is slow, if there is not a preallocated array for the output. Allocating an arrow for the esv might be a good idea.

# get logging levels
loglevel_debug_RC = log.getLevelName("DEBUG_RC")

loglevel_info_single_run = log.getLevelName("INFO_1RUN")
loglevel_info_multiple_run = log.getLevelName("INFO_nRUN")


class ReservoirComputer(object):
    """
    Class to model dynamics using a single reservoir.
    The class provides a function to train the reservoirs redout, propagate the reservoir state or perform a prediction.
    Further, private methods exist to collect the reservoir state or construct the input and adjacency matrix.
    """

    input_matrix: np.ndarray
    input_from_bias: np.ndarray
    adjacency_matrix: np.ndarray | scipy.sparse.coo_matrix
    output_ridgemodel: Ridge
    nodes: int
    degree: int
    leakage: float
    modified_leakage: float
    spectral_radius: float
    input_scaling: float
    input_length: int
    input_bias: float
    output_bias: float
    regularization: float
    training_includeinput: bool
    esv_lenght: int

    def __init__(
        self,
        nodes: int,
        degree: int,
        leakage: float,
        spectral_radius: float,
        input_scaling: float,
        input_length: int,
        input_bias: float,
        output_bias: float,
        regularization: float,
        training_includeinput: bool = True,
        identical_inputmatrix: bool = False,
        identical_adjacencymatrix: bool = False,
        identical_outputmatrix: bool | tuple | str = False,
        reset_matrices: bool = True,
    ) -> None:
        """
        Initializes a single reservoir based on an echo state network.

        Args:
            nodes (int):
                amount of nodes within the reservoir
            degree (int):
                degree of the adjacency matrix of the echo-state network
            leakage (float):
                describes how fast the reservoir forgets old data
                (0 fixed reservoir state, 1 no memory only driven by inputs)
            spectral_radius (float):
                spectral radius of the adjacency matrix
            input_scaling (float):
                scaling of the input data or weights
            input_length (int):
                length of the input data
            input_bias (float):
                bias added to the input data
            output_bias (float):
                bias added to the output data
            regularization (float):
                regularization parameter for the ridge regression
            training_includeinput (bool):
                if True the input data is included in the extended state vector, i.e. used in the ridge regression
            identical_inputmatrix (bool):
                if True the input matrix is the same for all :code:`ReservoirComputer` instances
            identical_adjacencymatrix (bool):
                if True the adjacency matrix is the same for all :code:`ReservoirComputer` instances
            identical_outputmatrix (bool | tuple | str):
                if False the output matrix is different for all :code:`ReservoirComputer` instances, else it is the same.
            reset_matrices (bool):
                if True the matrices are reset, else they are kept if they are identical for all parallel reservoirs.
        """
        # set reservoir class variables
        ReservoirComputer.nodes = nodes
        ReservoirComputer.degree = degree
        ReservoirComputer.leakage = leakage
        ReservoirComputer.modified_leakage = 1.0 - leakage
        ReservoirComputer.spectral_radius = spectral_radius

        ReservoirComputer.input_scaling = input_scaling
        ReservoirComputer.input_length = input_length
        ReservoirComputer.input_bias = input_bias

        ReservoirComputer.output_bias = output_bias
        ReservoirComputer.regularization = regularization
        ReservoirComputer.training_includeinput = training_includeinput

        if training_includeinput:
            ReservoirComputer.esv_lenght = nodes + input_length + 1
        else:
            ReservoirComputer.esv_lenght = nodes + 1
        self.esv = np.empty((1, ReservoirComputer.esv_lenght))
        self.esv[0, -1] = self.output_bias

        # deprecated variables (just in case)
        # self.nodes_effective = self.nodes + 1
        # self.state_expand = state_expand
        # self.dtype = dtype

        # initialize output matrix, untrained
        # either use one for all or individual Ridge instances
        if (isinstance(identical_outputmatrix, (tuple, str))) and (
            not hasattr(self, "output_ridgemodel") or reset_matrices
        ):
            # only if not done before set the class variable
            ReservoirComputer.output_ridgemodel = Ridge(alpha=regularization)
        elif identical_outputmatrix is False:
            self.output_ridgemodel = Ridge(alpha=regularization)
        elif (not isinstance(identical_outputmatrix, (tuple, str))) and (
            identical_outputmatrix is not False
        ):
            raise ValueError(
                f"Identical output matrix must be False, a tuple specifying the training or 'Combine', but is {identical_outputmatrix}."
            )

        # set input matrix & the contribution of the input bias
        # either use one for all or individual input matrices
        if identical_inputmatrix and (
            not hasattr(self, "input_matrix") or reset_matrices
        ):
            # only if not done before set the class variable
            (
                ReservoirComputer.input_matrix,
                ReservoirComputer.input_from_bias,
            ) = self._get_input_matrix()
        elif not identical_inputmatrix:
            self.input_matrix, self.input_from_bias = self._get_input_matrix()

        # set adjacency matrix
        # either use one for all or individual adjacency matrices
        if identical_adjacencymatrix and (
            not hasattr(self, "adjacency_matrix") or reset_matrices
        ):
            # only if not done before set the class variable
            ReservoirComputer.adjacency_matrix = self._get_adjacency_matrix()
        elif not identical_adjacencymatrix:
            self.adjacency_matrix = self._get_adjacency_matrix()

        # initialise memory for reservoir state with random values in [0, 1)
        # TODO: If RC is used without ParallelReservoirs, this is without seeds!
        self.state = np.random.rand(self.nodes)

    def train(
        self,
        input: np.ndarray,
        output: np.ndarray,
        transient_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Train a reservoir with given training data :code:`input, output` of shape :code:`(training_steps+transient_steps, variables_in), (training_steps, variables_out)`, respectively.

        For a training input time series :math:`\vec{u}(t)` the reservoir is updated and its states :math:`\vec{r}(t)` are collected in the extended interior state vector :math:`\vec{x}(t)=[\vec{r}(t), \vec{u}(t), b_{\mathrm{in}}]`.

        The output matrix :math:`\boldsymbol{W}_{\mathrm{out}}` is trained to preform :math:`\vec{y}(t)=\boldsymbol{W}_{\mathrm{out}}\vec{x}(t)`, where :math:`\vec{y}(t)` is the desired output time series.
        The training uses Tikhonov regularisation:

        .. math::

            \boldsymbol{W}_{\mathrm{out}}=\boldsymbol{Y}\boldsymbol{X}^T\left(\boldsymbol{X}\boldsymbol{X}^T+\beta\boldsymbol{I}\right)^{-1}


        , which analytically minimizes

        .. math::

            \sum_m\|\vec{y}^{\mathrm{true}}(m)-\boldsymbol{W}_{\mathrm{out}}\vec{x}_m\|^2+\beta\|\boldsymbol{W}_{\mathrm{out}}\|_2^2\quad.

        The output matrix :math:`\boldsymbol{W}_{\mathrm{out}}` is saved as a ridge regression model :code:`sklearn.linear_model.Ridge` and the data on which the fit is performed (training data and reservoir states) is returned for testing purposes.

        Args:
            np.ndarray[float]:
                Input data following shape as :code:`(time_steps, variables_in)`, where :code:`time_steps` is the number of timesteps (including transient steps) and :code:`variables_in` is the number of input variables
            np.ndarray[float]:
                Output data following shape as :code:`(time_steps, variables_out)`, where :code:`time_steps` is the number of timesteps (excluding transient steps) and :code:`variables_out` is the number of output variables
            transient_steps (int):
                Number of transient steps to adjust the reservoir state to the input data. These reservoir states are not used in the fit.

        Returns:
            tuple[np.ndarray[float]]:
                Time series of the extended interior state vector :math:`\vec{x}(t)`  and corresponding output data :math:`\vec{y}(t)` used for training.

        """

        if input.shape[0] != output.shape[0] + transient_steps:
            raise ValueError(
                f"Input and output training data must have different length (only input includes transient)."
            )

        # transient/ warmup: adjust reservoir state to input data
        for input_t in input[:transient_steps]:
            self.propagate_reservoir_state(input_t)

        # init extended interior state vector
        esv = np.empty((output.shape[0], self.esv_lenght))
        if self.training_includeinput:
            esv[:, self.nodes : -1] = input[transient_steps:]
        esv[:, -1] = self.output_bias
        esv[:, : self.nodes] = self._collect_reservoir_state(input[transient_steps:])
        # Symmetry breaking
        esv[:, : self.nodes // 2] *= esv[:, : self.nodes // 2]

        logger.log(
            loglevel_debug_RC,
            f"ReservoirComputer.train\t\t\t\tTraining esv shape: {esv.shape}",
        )
        # fit ridge regression model
        self.output_ridgemodel.fit(esv, output)

        # return training data for testing purposes
        return esv, output

    def train_on_multiple_datasets(
        self,
        inputs: list[np.ndarray],
        outputs: list[np.ndarray],
        transient_steps,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Train a reservoir on lists of with given training data :code:`inputs, outputs`. Where the :code:`i`-th list entry is of shape :code:`(training_steps_i+transient_steps, variables_in), (training_steps_i, variables_out)`, respectively.

        For each element in :code:`inputs`, i.e. training time series :math:`\vec{u}(t)` the reservoir is updated and its states :math:`\vec{r}(t)` are collected in the extended interior state vector :math:`\vec{x}(t)=[\vec{r}(t), \vec{u}(t), b_{\mathrm{in}}]`.

        The output matrix :math:`\boldsymbol{W}_{\mathrm{out}}` is trained to preform :math:`\vec{y}(t)=\boldsymbol{W}_{\mathrm{out}}\vec{x}(t)` for all elements in the lists :code:`inputs, outputs`, where :math:`\vec{y}(t)` is the desired output time series.
        The training uses Tikhonov regularisation:

        .. math::

            \boldsymbol{W}_{\mathrm{out}}=\boldsymbol{Y}\boldsymbol{X}^T\left(\boldsymbol{X}\boldsymbol{X}^T+\beta\boldsymbol{I}\right)^{-1}


        , which analytically minimizes

        .. math::

            \sum_m\|\vec{y}^{\mathrm{true}}(m)-\boldsymbol{W}_{\mathrm{out}}\vec{x}_m\|^2+\beta\|\boldsymbol{W}_{\mathrm{out}}\|_2^2\quad.

        The output matrix :math:`\boldsymbol{W}_{\mathrm{out}}` is saved as a ridge regression model :code:`sklearn.linear_model.Ridge` and the data on which the fit is performed (training data and reservoir states) is returned for testing purposes.

        Args:
            inputs (list[np.ndarray[float]]):
                List of input training data, each following the shape :code:`(time_steps, variables_in)`, where :code:`time_steps` is the number of timesteps (including transient steps) and :code:`variables_in` is the number of input variables
            outputs (list[np.ndarray[float]]):
                List of training output data, each following the shape :code:`(time_steps, variables_out)`, where :code:`time_steps` is the number of timesteps (excluding transient steps) and :code:`variables_out` is the number of output variables
            transient_steps (int):
                Number of transient steps to adjust the reservoir state to the input data. These reservoir states are not used in the fit.        Returns:

        Return:
            tuple[np.ndarray[float]]:
                Over all list elements :code:`inputs, outputs` appended time series of the extended interior state vector :math:`\vec{x}(t)`  and corresponding output data :math:`\vec{y}(t)` used for training.

        """

        # Testing and gethering length of training samples
        if len(inputs) != len(outputs):
            raise ValueError(
                f"Input and output training data must have the same length."
            )
        for input, output in zip(inputs, outputs):
            if input.shape[0] != output.shape[0] + transient_steps:
                raise ValueError(
                    f"Input and output training data must have different length (only input includes transient)."
                )
            if input.shape[1] != inputs[0].shape[1]:
                raise ValueError(
                    f"All input data must have the same number of variables."
                )

        # get number of samples and cumulative samples for indexing
        samples = [output.shape[0] for output in outputs]
        cum_samples = np.append([0], np.cumsum(samples, dtype=int))

        # init extended interior state vector
        esv = np.empty((np.sum(samples), self.esv_lenght))
        if self.training_includeinput:
            for i, input_data in enumerate(inputs):
                esv[cum_samples[i] : cum_samples[i + 1], self.nodes : -1] = input_data[
                    transient_steps:
                ]
        esv[:, -1] = self.output_bias

        # collect reservoir states
        for i, input_data in enumerate(inputs):
            for input_t in input_data[:transient_steps]:
                self.propagate_reservoir_state(input_t)
            esv[
                cum_samples[i] : cum_samples[i + 1], : self.nodes
            ] = self._collect_reservoir_state(input_data[transient_steps:])

        # Symmetry breaking
        esv[:, : self.nodes // 2] *= esv[:, : self.nodes // 2]

        # fit ridge regression model
        output_concatenated = np.concatenate(outputs, axis=0)
        self.output_ridgemodel.fit(esv, output_concatenated)

        return esv, output_concatenated

    def predict_single_step(self, input: np.ndarray) -> np.ndarray:
        r"""Predict a single time step.

        Uses the input data to updata the reservoir state following

        .. math::

            \vec{s}(t+1) = (1-\alpha) \vec{s}(t) + \alpha \tanh(\nu \boldsymbol{W}_{\mathrm{in}} [\vec{u}(t), b_{\mathrm{in}}] + \rho \boldsymbol{W} s(t))

        and predict the following time step with

        .. math::

            \vec{u}(t + \Delta t) = \boldsymbol{W}_{\mathrm{out}} [\vec{s}(t), \vec{u}(t), b_{\mathrm{out}}] \,.

        Args:
            np.ndarray[float]:
                Input data containing one timestep, of shape: :code:`(1, ReservoirComputer.input_length)`.

        Returns:
            np.ndarray[float]:
                Predicted output data for the next timestep, of shape :code:`(1, ReservoirComputer.input_length)`.

        Warning:
            This function uses an hstack, which is slow, if there is not a preallocated array for the output. Allocating an arrow for the esv might be a good idea.
        """
        logger.log(
            loglevel_debug_RC,
            f"ReservoirComputer.predict_single_step\t\tInput shape: {input.shape}",
        )

        # propagate reservoir state
        self.propagate_reservoir_state(input[0])

        # create extended state vector
        self.esv[0, : self.nodes] = self.state[: self.nodes]
        self.esv[0, : self.nodes // 2] *= self.esv[
            0, : self.nodes // 2
        ]  # Symmetry breaking
        if self.training_includeinput:
            self.esv[0, self.nodes : -1] = input[0]
        # last component is not changed always set to output_bias

        # predict output
        return self.output_ridgemodel.predict(self.esv)

    def propagate_reservoir_state(
        self,
        input: np.ndarray,
    ) -> None:
        r"""
        Propagates a reservoir state to the next timestep using an :code:`input` time series  :math:`\vec{u}(t)` following

        .. math::

            \vec{s}(t+1) = (1-\alpha) \vec{s}(t) + \alpha \tanh(\nu \boldsymbol{W}_{\mathrm{in}} [\vec{u}(t), b_{\mathrm{in}}] + \rho \boldsymbol{W} s(t))

        Args:
            np.ndarray[float]:
                Input (transient) time series :math:`\vec{u}(t)` of shape :code:`(variables,)`.

        Notes:
            This function returns a reference, hence modifying the return value directly modifies the reseroir state.

        """
        self.state *= self.modified_leakage
        self.state += self.leakage * np.tanh(
            self.adjacency_matrix @ self.state
            + self.input_matrix @ input
            + self.input_from_bias
        )

    def _collect_reservoir_state(self, input_data: np.ndarray) -> np.ndarray:
        r"""
        Collects the reservoir states, which are updated following

        .. math::

            \vec{s}(t+1) = (1-\alpha) \vec{s}(t) + \alpha \tanh(\nu \boldsymbol{W}_{\mathrm{in}} [\vec{u}(t), b_{\mathrm{in}}] + \rho \boldsymbol{W} s(t))

        for a given input :code:`input_data`, i.e. a time series :math:`\vec{u}(t)`.

        Args:
            input_data (np.ndarray[float]):
                Input time series :math:`\vec{u}(t)` of shape :code:`(time_steps, variables)`.
        Returns:
            np.ndarray[float]:
                Reservoir time series :math:`\vec{s}(t)` of shape :code:`(time_steps, self.nodes)`.
        """
        collected_states = np.empty((input_data.shape[0], self.nodes))
        for t, input_t in enumerate(input_data):
            self.propagate_reservoir_state(input_t)
            collected_states[t] = self.state
        return collected_states

    def _get_input_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        r"""Return :math:`\boldsymbol{W}_{\mathrm{in}}`, which maps the input to reservoir nodes.

        Use the shape of input data and nnumber of nodes to determine the matrix.

        Returns the input matrix for the input data and separately the input
        contribution of the input bias
        """
        input_matrix = self.input_scaling * np.random.uniform(
            low=-0.5, high=0.5, size=(self.nodes, self.input_length)
        )
        input_from_bias = (
            self.input_scaling
            * self.input_bias
            * np.random.uniform(low=-0.5, high=0.5, size=(self.nodes))
        )
        return input_matrix, input_from_bias

    def _get_adjacency_matrix(self) -> np.ndarray | scipy.sparse.coo_matrix:
        r"""Returns the adjacency matrix :math:`\boldsymbol{W}` of the reservoir. This is the cuppeling matrix of the inner nodes.
        The spectral radius of the adjacency matrix is set to one and is multiplied in the update step :code:`propagate_reservoir_state`.
        """
        # set adjacency matrix, as degree is low compared to dimension use sparse matrix
        adjacency = scipy.sparse.random(
            m=self.nodes,
            n=self.nodes,
            density=self.degree / self.nodes,
        )
        # ensure spectral radius to be one
        max_eigenvalue = np.abs(
            eigs(adjacency, k=1, which="LM", return_eigenvectors=False)
        )

        return (self.spectral_radius / max_eigenvalue[0]) * adjacency
