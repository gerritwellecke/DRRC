import logging as log
from typing import List, cast

import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.metrics import mean_squared_error

from drrc.config import Config
from drrc.reservoircomputer import ReservoirComputer

log_debug_testlevel = 18
log_info_testlevel = 19


def test_singleRC_1d() -> None:
    """Test the training of a ReservoirComputer class."""

    # Fix seed to always get identical results
    np.random.seed(seed=0)

    # Define length
    traing_steps = 150
    transient_steps = 100
    eval_steps = 150

    # Init ReservoirComputer
    res = ReservoirComputer(
        nodes=100,
        degree=4,
        leakage=0.9,
        spectral_radius=1,
        input_scaling=1,
        input_length=1,
        input_bias=1,
        output_bias=1,
        regularization=0.00001,
        training_includeinput=False,
        identical_inputmatrix=True,
        identical_outputmatrix=False,
    )

    # Test constant predictions of RC
    training_data = np.ones((traing_steps, 1))
    eval_data = np.ones((eval_steps, 1))
    _test_prediction(
        res,
        training_data[:-1],
        training_data[1:],
        eval_data,
        transient_steps,
        Name="1D_Constant",
    )

    # Test alternating predictions of RC
    training_data[::2] = -1
    eval_data[::2] = -1
    _test_prediction(
        res,
        training_data[:-1],
        training_data[1:],
        eval_data,
        transient_steps,
        Name="1D_Alternating",
    )

    # test some sinusoidal data
    training_steps = 10000
    t = np.linspace(0, 20 * 2 * np.pi, traing_steps)
    training_data = (np.sin(t)).reshape(-1, 1)
    eval_data = (np.sin(t[:eval_steps])).reshape(-1, 1)
    _test_prediction(
        res,
        training_data[:-1],
        training_data[1:],
        eval_data,
        transient_steps,
        Name="1D_Sinusoidal",
    )

    # TODO: here should something more complex be tested


def _test_prediction(
    res: ReservoirComputer,
    training_data_in: np.ndarray,
    training_data_out: np.ndarray,
    eval_data: np.ndarray,
    transient_steps: int,
    Name: str,
) -> None:
    """Test the prediction of a ReservoirComputer class.

    Args:
        res (ReservoirComputer): ReservoirComputer class.
        training_data_in (np.ndarray): Training data input.
        training_data_out (np.ndarray): Training data output.
        eval_data (np.ndarray): Evaluation data.
        transient_steps (int): Number of transient steps.
        Name (str): Name of the test, will be file name.

    Returns:
        None

    """
    # Test Training
    ## Train ReservoirComputer
    log.log(log_debug_testlevel, f"{Name}:\tTraining.")
    esv, output = res.train(
        training_data_in, training_data_out[transient_steps:], transient_steps
    )
    train_predict = res.output_ridgemodel.predict(esv)
    rmse = np.sqrt(mean_squared_error(train_predict, output))

    ## Generate Output regarding prediction
    log.log(log_info_testlevel, f"Training RMSE for {Name} is: {rmse}")
    log.log(log_debug_testlevel, f"{Name}:\tGenerating training output.")
    _plot(
        truth=np.vstack((training_data_out[:transient_steps], output)),
        prediction=np.vstack(
            (np.zeros((transient_steps, train_predict.shape[1])), train_predict)
        ),
        transient_steps=transient_steps,
        name=f"{Name}_Training",
    )
    assert rmse < 1e-1, f"Training RMSE for {Name} is too large. It is {rmse}."

    # Test predictions
    log.log(log_debug_testlevel, f"{Name}:\tTest predicitons of transients.")
    pred = np.zeros((len(eval_data), eval_data.shape[1]))
    ## Transient steps
    for t in range(transient_steps):
        # print(eval_data[t:t+1][0], res.state[0])
        pred[t] = res.predict_single_step(eval_data[t : t + 1])[0]
    state_after_transient = np.copy(res.state)

    log.log(
        log_debug_testlevel, f"{Name}:\tTest iterative predicitons after transient."
    )
    ## Prediction
    for t in range(transient_steps, len(eval_data)):
        # print(eval_data[t:t+1][-1])
        pred[t] = res.predict_single_step(pred[t - 1 : t])[0]

    ## Generate Output regarding prediction
    log.log(log_debug_testlevel, f"{Name}:\tGenerating prediction output.")
    _plot(
        truth=eval_data[1:],
        prediction=pred[:-1],
        transient_steps=transient_steps,
        name=f"{Name}_Prediction_withTransientPrediciton",
    )
    for t in range(transient_steps, len(eval_data) - 1):
        assert np.isclose(
            np.linalg.norm(pred[t] - eval_data[t + 1]), 0, atol=0.1
        ), f"Prediction error for {Name} prediction at prediction step {t-transient_steps} is too large. It is {np.linalg.norm(pred[t]-eval_data[t+1])}"

    # Test if transient step methods are equal
    log.log(log_debug_testlevel, f"{Name}:\tPropagate state without prediction.")
    for eval_data_t in eval_data[:transient_steps]:
        res.propagate_reservoir_state(eval_data_t)
    # print(state.shape, state_after_transient.shape)
    # print(state, state_after_transient)
    assert np.isclose(
        np.linalg.norm(res.state - state_after_transient), 0, atol=0.1
    ), f"Transient step methods are not equal for {Name}. 2-Norm between states after different transient methods is {np.linalg.norm(res.state-state_after_transient)}"


def _plot(
    truth: np.ndarray, prediction: np.ndarray, transient_steps: int, name: str
) -> None:
    """
    Plot the truth, prediction, and error.

    Args:
        truth (np.ndarray): Ground truth values.
        prediction (np.ndarray): Predicted values.
        transient_steps (int): Number of transient steps.
        name (str): Name of the plot.

    Returns:
        None
    """
    if truth.shape[1] < 10:
        fig, axs = plt.subplots(truth.shape[1])
        error = np.abs(truth - prediction)
        vmin, vmax = np.min(np.hstack((truth, prediction, error))), np.max(
            np.hstack((truth, prediction, error))
        )
        if truth.shape[1] == 1:
            axs = cast(
                List[plt.Axes], [axs]
            )  # this is complecated only to help the type checking
        for ax in axs:
            ax.plot(truth, label="Truth", color="green")
            ax.plot(prediction, label="Prediction", color="blue")
            ax.plot(error, label="Error", color="red")
            ax.vlines(
                transient_steps,
                vmin,
                vmax,
                color="black",
                linestyles="dashed",
                label=" Prediction Start",
            )

            ax.set_xticks([transient_steps, len(truth)])
            ax.set_xticklabels([f"{transient_steps}", f"{len(truth)}"])

            ax.text(
                transient_steps / 2,
                float(np.mean(ax.get_ylim())),
                "Transient",
                ha="center",
                va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
            )
            ax.text(
                transient_steps + 3,
                float(np.mean(ax.get_ylim())),
                r"Prediction $\rightarrow$",
                ha="left",
                va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
            )

            ax.set_xlabel("Steps")
            ax.set_ylabel("Value")
    else:
        raise NotImplementedError(
            "Plotting for more than 10 dimensions is not implemented."
        )
        # should be imshow or something similar
    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=len(labels),
    )
    plt.savefig(f"{Config.get_git_root()}/Tests/TestResults/Test_{name}.png")
    plt.close()


if __name__ == "__main__":
    log.basicConfig(level=log_debug_testlevel)

    test_singleRC_1d()
    print("All tests passed.")
