import pickle
import numpy as np
import time

from agimus_controller_examples.utils.set_models_and_mpc import get_mpc
from agimus_controller.agimus_controller_examples.read_from_bag_trajectory import (
    save_rosbag_inputs_to_pickle,
)
from agimus_controller.trajectory import TrajectoryPoint


def find_array_nearest_idx(array, value):
    temp = abs(array - value)
    idx = np.where((temp == temp.min()))[0][0]
    if array[idx] > value:
        idx -= 1
    return idx


if __name__ == "__main__":
    # Get Inputs data
    bag_file_folder = "/tmp/rosbag_mpc_data_7"
    bag_file_name = "rosbag_mpc_data_7_0.db3"
    bag_file_path = bag_file_folder + "/" + bag_file_name
    input_picke_file_path = bag_file_folder + "/pickle_input_" + bag_file_name
    save_rosbag_inputs_to_pickle(bag_file_path, input_picke_file_path)
    with open(input_picke_file_path, "rb") as pickle_file:
        mpc_inputs_data = pickle.load(pickle_file)
    mpc_inputs = mpc_inputs_data["mpc_inputs"]
    x0s = mpc_inputs_data["x0s"]
    x0s_timestamp = np.array(mpc_inputs_data["x0s_timestamp"])
    mpc_inputs_timestamp = np.array(mpc_inputs_data["mpc_inputs_timestamp"])

    start_idx_x0 = find_array_nearest_idx(mpc_inputs_timestamp, x0s_timestamp[0])
    mpc = get_mpc()
    mpc_data = {}
    mpc_data["state_references"] = []
    mpc_data["control_references"] = []
    mpc_data["ee_pose_references"] = []
    mpc_data["states_predictions"] = []
    mpc_data["control_predictions"] = []
    mpc_data["collision_distance_residuals"] = []
    mpc_data["kkt_norm"] = []
    mpc_data["solve_time"] = []
    mpc_data["nb_iter"] = []
    mpc_data["nb_qp_iter"] = []

    for idx in range(len(mpc_inputs)):
        w_traj_point = mpc_inputs[idx]
        traj_point = w_traj_point.point
        mpc.append_trajectory_point(w_traj_point)
        mpc_input_timestamp = traj_point.time_ns
        mpc_data["state_references"].append(
            np.concatenate((traj_point.robot_configuration, traj_point.robot_velocity))
        )
        mpc_data["control_references"].append(traj_point.robot_effort)
        pos = list(traj_point.end_effector_poses.values())[0].translation
        mpc_data["ee_pose_references"].append(pos)
        if idx >= start_idx_x0:
            start_time = time.perf_counter()
            x0 = x0s[idx - start_idx_x0]
            x0_traj_point = TrajectoryPoint(
                time_ns=x0s_timestamp[idx - start_idx_x0],
                robot_configuration=x0.joint_state.position,
                robot_velocity=x0.joint_state.velocity,
                robot_acceleration=np.zeros_like(x0.joint_state.velocity),
            )
            ocp_res = mpc.run(
                initial_state=x0_traj_point,
                current_time_ns=x0s_timestamp[idx - start_idx_x0],
            )
            mpc_debug_data = mpc.mpc_debug_data

            mpc_data["states_predictions"].append(ocp_res.states)
            mpc_data["control_predictions"].append(ocp_res.feed_forward_terms)
            mpc_data["collision_distance_residuals"].append(
                np.array(mpc_debug_data.ocp.collision_distance_residuals)
            )
            mpc_data["kkt_norm"].append(mpc_debug_data.ocp.kkt_norm)
            mpc_data["nb_iter"].append(mpc_debug_data.ocp.iter)
            mpc_data["nb_qp_iter"].append(mpc_debug_data.ocp.qp_iters)
            solve_time = time.perf_counter() - start_time
            mpc_data["solve_time"].append(solve_time)
    output_picke_file_path = bag_file_folder + "/pickle_replayed_" + bag_file_name
    with open(output_picke_file_path, "wb") as pickle_file:
        pickle.dump(mpc_data, pickle_file)
