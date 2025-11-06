import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from typing import Union
import pinocchio as pin

from agimus_controller.plots.plot_tails import plot_tails


def plot_values(
    title: str,
    values_array: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
    labels: Union[None, list[str]] = None,
    ylabels: Union[None, list[str]] = None,
    semilogs: Union[None, list[bool]] = None,
    ylimits: Union[None, list[list[float]]] = None,
) -> None:
    """Subplots from concatenated array of values."""
    values_array = np.array(values_array)
    if len(values_array.shape) == 1:
        values_array = values_array[:, np.newaxis]
    nb_plots = values_array.shape[1]
    fig, ax = plt.subplots(nb_plots, 1)
    fig.canvas.manager.set_window_title(title)
    if nb_plots == 1:
        if labels is not None:
            ax.plot(time, values_array, label=labels[0])
        else:
            ax.plot(time, values_array)
        ax.legend()
        ax.set_xlabel("t (s)")
        if ylabels is not None:
            ax.set_ylabel(ylabels)
    else:
        for i in range(values_array.shape[1]):
            if labels is not None:
                if semilogs is not None and semilogs[i] is True:
                    ax[i].semilogy(time, values_array[:, i], label=labels[i])
                else:
                    ax[i].plot(time, values_array[:, i], label=labels[i])
            else:
                ax[i].plot(time, values_array[:, i])
            ax[i].legend()
            ax[i].set_xlabel("t (s)")
            if ylimits is not None:
                ax[i].set_ylim(ylimits[i][0], ylimits[i][1])
            if ylabels is not None:
                ax[i].set_ylabel(ylabels)


def plot_values_on_same_fig(
    title: str,
    values_array: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
    labels: Union[None, list[str]] = None,
) -> None:
    """Plot concatenated array of values on the same figure."""
    values_array = np.array(values_array)
    fig, ax = plt.subplots(1, 1)
    fig.canvas.manager.set_window_title(title)
    for i in range(values_array.shape[1]):
        if labels is not None:
            ax.plot(time, values_array[:, i], label=labels[i])
        else:
            ax.plot(time, values_array[:, i])
        ax.legend()
        ax.set_xlabel("t (s)")


def concatenate_arrays_columns(
    array1: npt.NDArray[np.float64], array2: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Concatenate array1 with array2 by columns."""
    if len(array1.shape) == len(array2.shape):
        array1 = array1[:, np.newaxis]
    return np.c_[array1[: array2.shape[0]], array2[:, np.newaxis]]


def concatenate_array_with_list_of_arrays(
    array: npt.NDArray[np.float64], list_array: list[npt.NDArray[np.float64]]
) -> npt.NDArray[np.float64]:
    """Concatenate array with given list of array by columns."""
    for array_to_concatenate in list_array:
        array = np.c_[
            array[: array_to_concatenate.shape[0]], array_to_concatenate[:, np.newaxis]
        ]
    return array


def plot_mpc_data(
    mpc_data: dict[str, npt.NDArray[np.float64]],
    mpc_config: dict[str, Union[int, float, str]],
    rmodel: pin.Model,
    which_plots: list[str],
) -> None:
    """Plots MPC data specified in which_plots list.

    Arguments:
        mpc_data: dictionary containing MPC data, keys are:
            - "solve_time": array of computation times for each MPC iteration
            - "distance": array of collision pairs distances
            - "kkt_norms": array of KKT norms for each MPC iteration
            - "nb_iters": array of number of iterations for each MPC iteration
            - "nb_qp_iters": array of number of QP iterations for each MPC iteration
            - "mpc_inputs": list of MPC input messages, each containing weights and references
            - "states_predictions": array of state predictions for each MPC iteration
            - "control_predictions": array of control predictions for each MPC iteration
            - "goal_tracking_references": array of goal tracking references for each MPC iteration
        rmodel: pinocchio model of the robot
        which_plots: list of strings specifying which plots to generate, options are:
            - "computation_time": plot computation times for each MPC iteration
            - "collision_distance": plot distances between collision pairs
            - "iter": plot KKT norms and number of iterations for each MPC iteration
            - "visual_servoing": plot visual servoing states
            - "predictions": plot predictions of states and controls for each MPC iteration
    """
    # Computation time plots
    if "computation_time" in which_plots:
        solve_time = np.array(mpc_data["solve_time"])
        time = np.linspace(0, (solve_time.shape[0] - 1) * 0.01, solve_time.shape[0])
        # plot_mpc_iter_durations("MPC iterations duration", solve_time, time)
        plot_values("MPC iterations duration", solve_time, time)
        print("solve time mean ", np.mean(solve_time))

    # Collisions pairs distance plots
    coll_avoidance_keys = [
        val for val in list(mpc_data.keys()) if "avoid_collision" in val
    ]
    if "collision_distance" in which_plots and not coll_avoidance_keys:
        print(
            "No collision pairs distances in mpc_data dictionary, ",
            "keys are ",
            mpc_data.keys(),
        )
    if "collision_distance" in which_plots and coll_avoidance_keys:
        coll_distance_residuals = []
        for key in coll_avoidance_keys:
            coll_distance_residuals.append(np.array(mpc_data[key])[:, 0])
        nb_vals = len(mpc_data[coll_avoidance_keys[0]])
        coll_distance_residuals = np.array(coll_distance_residuals)
        coll_distance_residuals = coll_distance_residuals.transpose()
        time_col = np.linspace(
            0,
            (nb_vals - 1) * 0.01,
            nb_vals,
        )
        coll_labels = [f"col_term_{i}" for i in range(coll_distance_residuals.shape[0])]
        plot_values(
            "collision pairs distances", coll_distance_residuals, time_col, coll_labels
        )

    # Number of iterations and kkt norms
    if "iter" in which_plots:
        kkt_norms = np.array(mpc_data["kkt_norms"])
        nb_iters = np.array(mpc_data["nb_iters"])
        nb_qp_iters = np.array(mpc_data["nb_qp_iters"])
        concatenated_values = concatenate_array_with_list_of_arrays(
            kkt_norms, [nb_iters, nb_qp_iters]
        )
        time = np.linspace(0, (kkt_norms.shape[0] - 1) * 0.01, kkt_norms.shape[0])

        plot_values(
            "kkt norm and solver iterations",
            concatenated_values,
            time,
            ["kkt norms ", "nb iterations", "nb qp iters"],
            semilogs=[True, False, False],
        )

    # Visual servoing
    if "visual_servoing" in which_plots:
        w_pose = [
            next(iter(mpc_input.weights.w_end_effector_poses.values()))
            for mpc_input in mpc_data["mpc_inputs"]
        ]

        visual_servoing_state = np.zeros((len(w_pose), 1))
        for idx in range(len(w_pose)):
            if (w_pose[idx] == np.zeros(6)).all():
                visual_servoing_state[idx] = 0
            elif w_pose[idx][0] >= w_pose[idx - 1][0]:
                visual_servoing_state[idx] = 1
            else:
                visual_servoing_state[idx] = 2
        time = np.linspace(
            0,
            (visual_servoing_state.shape[0] - 1) * 0.01,
            visual_servoing_state.shape[0],
        )
        plot_values(
            "Visual servoing state",
            visual_servoing_state,
            time,
            [
                "0 : IDLE, 1: VISUAL_SERVOING_ACTIVE, 2: COMING_BACK_TO_IDLE",
            ],
        )

    # Plot predictions
    if "predictions" in which_plots:
        mpc_xs = np.array(mpc_data["states_predictions"])
        mpc_us = np.array(mpc_data["control_predictions"])

        ctrl_refs = (
            np.array(mpc_data["control_reg_references"])
            if "control_reg_references" in mpc_data.keys()
            else None
        )
        state_refs = (
            np.array(mpc_data["state_reg_references"])
            if "state_reg_references" in mpc_data.keys()
            else None
        )
        translation_refs = (
            np.array(mpc_data["goal_tracking_references"])[:, :3]
            if "goal_tracking_references" in mpc_data.keys()
            else None
        )
        plot_tails(
            mpc_xs,
            mpc_us,
            rmodel,
            mpc_config=mpc_config,
            ctrl_refs=ctrl_refs,
            state_refs=state_refs,
            translation_refs=translation_refs,
        )
