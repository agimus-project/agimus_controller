from rclpy.serialization import deserialize_message
import rosbag2_py
import numpy as np
import pickle
from builtin_interfaces.msg import Duration
from linear_feedback_controller_msgs_py.numpy_conversions import matrix_msg_to_numpy
from agimus_controller_ros.ros_utils import mpc_msg_to_weighted_traj_point
from agimus_msgs.msg import MpcInput, MpcDebug
from linear_feedback_controller_msgs.msg import Sensor
from linear_feedback_controller_msgs_py.numpy_conversions import sensor_msg_to_numpy


def convert_bytes_to_message(serialized_bytes, msg_type):
    """Deserialize the bytes into the correct message type."""
    message = deserialize_message(serialized_bytes, msg_type)
    return message


# Define function to convert ROS 2 messages to a pickle file
def save_rosbag_inputs_to_pickle(bag_file_path, pickle_file_path):
    # Open the rosbag
    storage_options = rosbag2_py.StorageOptions(uri=bag_file_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # Open the pickle file
    with open(pickle_file_path, "wb") as pickle_file:
        mpc_inputs_data = {}
        mpc_inputs_data["mpc_inputs"] = []
        mpc_inputs_data["x0s"] = []
        mpc_inputs_data["x0s_timestamp"] = []
        mpc_inputs_data["mpc_inputs_timestamp"] = []
        # Read each message in the rosbag and store it
        while reader.has_next():
            topic, msg, timestamp = reader.read_next()
            if topic == "/mpc_input":
                mpc_inputs_data["mpc_inputs"].append(
                    mpc_msg_to_weighted_traj_point(
                        convert_bytes_to_message(msg, MpcInput), timestamp
                    )
                )
                mpc_inputs_data["mpc_inputs_timestamp"].append(timestamp)
            elif topic == "/ocp_x0":
                mpc_inputs_data["x0s"].append(
                    sensor_msg_to_numpy(convert_bytes_to_message(msg, Sensor))
                )
                mpc_inputs_data["x0s_timestamp"].append(timestamp)
        # Serialize data to pickle file
        pickle.dump(mpc_inputs_data, pickle_file)


# Define function to convert ROS 2 messages to a pickle file
def save_rosbag_outputs_to_pickle(bag_file_path, pickle_file_path):
    # Open the rosbag
    storage_options = rosbag2_py.StorageOptions(uri=bag_file_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # Open the pickle file
    with open(pickle_file_path, "wb") as pickle_file:
        mpc_data = {}
        mpc_data["state_references"] = []
        mpc_data["control_references"] = []
        mpc_data["ee_pose_references"] = []
        mpc_data["states_predictions"] = []
        mpc_data["control_predictions"] = []
        mpc_data["collision_distance_residuals"] = []
        mpc_data["kkt_norms"] = []
        mpc_data["nb_iters"] = []
        mpc_data["nb_qp_iters"] = []
        mpc_data["solve_time"] = []

        # Read each message in the rosbag and store it
        while reader.has_next():
            topic, msg, _ = reader.read_next()
            if topic == "/mpc_input":
                msg = convert_bytes_to_message(msg, MpcInput)
                state_reference = np.concatenate((msg.q, msg.qdot))
                mpc_data["state_references"].append(state_reference)
                mpc_data["control_references"].append(np.array(msg.robot_effort))
                pos = msg.pose.position
                mpc_data["ee_pose_references"].append(np.array([pos.x, pos.y, pos.z]))
            elif topic == "/mpc_debug":
                mpc_debug_msg = convert_bytes_to_message(msg, MpcDebug)
                mpc_data["states_predictions"].append(
                    matrix_msg_to_numpy(mpc_debug_msg.states_predictions)
                )
                mpc_data["control_predictions"].append(
                    matrix_msg_to_numpy(mpc_debug_msg.control_predictions)
                )
                mpc_data["collision_distance_residuals"].append(
                    matrix_msg_to_numpy(mpc_debug_msg.collision_distance_residuals)
                )
                mpc_data["kkt_norms"].append(mpc_debug_msg.kkt_norm)
                mpc_data["nb_iters"].append(mpc_debug_msg.nb_iter)
                mpc_data["nb_qp_iters"].append(mpc_debug_msg.nb_qp_iter)
            elif topic == "/ocp_solve_time":
                solve_time_msg = convert_bytes_to_message(msg, Duration)
                mpc_data["solve_time"].append(
                    solve_time_msg.sec + 1e-9 * solve_time_msg.nanosec
                )
        # Serialize data to pickle file
        pickle.dump(mpc_data, pickle_file)


# Example usage
if __name__ == "__main__":
    bag_file = "/home/gepetto/ros2_ws/src/agimus_controller/bag_files/slow_sin"  # Path to the bag file without the .db3 extension
    pickle_file = "slow_sim_weighted_trajectory_data.pkl"  # Path to the pickle file

    save_rosbag_outputs_to_pickle(bag_file, pickle_file)
    print(f"Saved ROS 2 bag data to {pickle_file}")
