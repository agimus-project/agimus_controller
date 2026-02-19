import numpy as np
import numpy.typing as npt
from typing import Union
import pinocchio as pin

from agimus_controller.plots.mpc_utils import (
    init_sim_data,
    extract_plot_data_from_sim_data,
    plot_mpc_results,
)


def get_active_costs(
    ctrl_refs: Union[None, npt.NDArray[np.float64]],
    state_refs: Union[None, npt.NDArray[np.float64]],
    translation_refs: Union[None, npt.NDArray[np.float64]],
) -> list[str]:
    """Returns list of active costs."""
    active_costs = []
    if ctrl_refs is not None:
        active_costs.append("ctrlReg")
    if state_refs is not None:
        active_costs.append("stateReg")
    if translation_refs is not None:
        active_costs.append("translation")
    return active_costs


def plot_tails(
    mpc_xs: npt.NDArray[np.float64],
    mpc_us: npt.NDArray[np.float64],
    model: pin.Model,
    mpc_config: dict[str, Union[int, float, str]],
    ctrl_refs: Union[None, npt.NDArray[np.float64]] = None,
    state_refs: Union[None, npt.NDArray[np.float64]] = None,
    translation_refs: Union[None, npt.NDArray[np.float64]] = None,
) -> None:
    """
    Plot tails from given mpc data.
    mpc_xs : numpy array storing the prediction xs at each iteration of the mpc
        shape : np.ndarray[number of iteration of your mpc, number of nodes, size of state vector]
    mpc_us : numpy array storing the prediction us at each iteration of the mpc
        shape : np.ndarray[number of iteration of your mpc, number of nodes-1, size of control vector]
    model : pinocchio model of the robot
    mpc_config : dictionary loaded from the mpc_config.yaml file

    Optional parameters:
    ctrl_refs : control reference of your first node at each mpc iteration,
        shape : np.ndarray[number of iteration of your mpc, size of control vector]
    state_refs : state reference of your first node at each mpc iteration,
        shape : np.ndarray[number of iteration of your mpc, size of state vector]
    translation_refs : translation reference control of your first node at each mpc iteration,
        shape : np.ndarray[number of iteration of your mpc, 3]
    """
    ocp_params = {}
    ocp_params["N_h"] = mpc_config["nb_running_nodes"]
    ocp_params["dt"] = mpc_config["dt_ocp"]
    ocp_params["pin_model"] = model
    ocp_params["id_endeff"] = model.getFrameId(mpc_config["endeff_name"])
    ocp_params["armature"] = np.zeros([model.nq])

    active_costs = get_active_costs(ctrl_refs, state_refs, translation_refs)
    ocp_params["active_costs"] = active_costs

    # Simu parameters
    sim_params = {}
    sim_params["sim_freq"] = mpc_config["mpc_freq"]
    sim_params["mpc_freq"] = mpc_config["mpc_freq"]
    if "T_sim" in mpc_config.keys():
        sim_params["T_sim"] = mpc_config["T_sim"]
    else:
        sim_params["T_sim"] = (mpc_xs.shape[0] - 1) / mpc_config["mpc_freq"]

    # Initialize simulation data
    sim_data = init_sim_data(sim_params, ocp_params, mpc_xs[0, 0, :])

    for mpc_cycle in range(sim_data["N_sim"]):
        sim_data["state_pred"][mpc_cycle, :, :] = mpc_xs[mpc_cycle, :, :]
        sim_data["ctrl_pred"][mpc_cycle, :, :] = mpc_us[mpc_cycle, :, :]
        # Extract relevant predictions for interpolations
        x_curr = sim_data["state_pred"][
            mpc_cycle, 0, :
        ]  # x0* = measured state    (q^,  v^ )
        x_pred = sim_data["state_pred"][
            mpc_cycle, 1, :
        ]  # x1* = predicted state   (q1*, v1*)
        u_curr = sim_data["ctrl_pred"][
            mpc_cycle, 0, :
        ]  # u0* = optimal control   (tau0*)
        # Record costs references
        if ctrl_refs is not None:
            sim_data["ctrl_ref"][mpc_cycle, :] = ctrl_refs[mpc_cycle, :]
        if state_refs is not None:
            sim_data["state_ref"][mpc_cycle, :] = state_refs[mpc_cycle, :]
        if translation_refs is not None:
            sim_data["lin_pos_ee_ref"][mpc_cycle, :] = translation_refs[mpc_cycle, :]

        # Select reference control and state for the current MPC cycle
        x_ref_MPC_RATE = x_curr + sim_data["ocp_to_mpc_ratio"] * (x_pred - x_curr)
        u_ref_MPC_RATE = u_curr
        if mpc_cycle == 0:
            sim_data["state_des_MPC_RATE"][mpc_cycle, :] = x_curr
        sim_data["ctrl_des_MPC_RATE"][mpc_cycle, :] = u_ref_MPC_RATE
        sim_data["state_des_MPC_RATE"][mpc_cycle + 1, :] = x_ref_MPC_RATE

        sim_data["state_mea_SIM_RATE"][mpc_cycle + 1, :] = mpc_xs[mpc_cycle + 1, 0, :]
    plot_data = extract_plot_data_from_sim_data(sim_data)
    dump_dict = plot_mpc_results(
        plot_data,
        which_plots=["x", "u", "ee"],
        PLOT_PREDICTIONS=True,
        pred_plot_sampling=int(sim_params["mpc_freq"] / 10),
        SAVE=True,
    )
    return dump_dict
