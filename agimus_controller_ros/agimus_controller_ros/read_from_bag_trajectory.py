# Standard library imports
import pickle
from pathlib import Path

# Third-party imports
import rosbag2_py
from rclpy.serialization import deserialize_message

# ROS message imports
from agimus_msgs.msg import MpcDebug, MpcInput
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from linear_feedback_controller_msgs.msg import Sensor
from linear_feedback_controller_msgs_py.numpy_conversions import (
    matrix_msg_to_numpy,
    sensor_msg_to_numpy,
)

# Local package imports
from agimus_controller_ros.ros_utils import mpc_msg_to_weighted_traj_point


def convert_bytes_to_message(serialized_bytes, msg_type):
    """Deserialize the bytes into the correct message type."""
    message = deserialize_message(serialized_bytes, msg_type)
    return message


def get_bag_storage_id(bag_folder_path):
    """Detect the storage format of a ROS 2 bag folder."""
    bag_path = Path(bag_folder_path)

    print("Detecting bag storage format...")
    # Check if it's a directory
    if bag_path.is_dir():
        # Look for .mcap files in the directory
        mcap_files = list(bag_path.glob("*.mcap"))
        if mcap_files:
            print("Detected MCAP storage format.")
            return "mcap"

        # Look for .db3 files in the directory
        db3_files = list(bag_path.glob("*.db3"))
        if db3_files:
            print("Detected SQLite3 storage format.")
            return "sqlite3"
    else:
        raise ValueError(f"The provided bag path {bag_folder_path} is not a directory.")

    # Default to sqlite3 for backward compatibility
    raise ValueError(
        f"Could not detect storage format in {bag_folder_path}. "
        "Ensure it contains .mcap or .db3 files."
    )


def load_mpc_inputs_from_rosbag(bag_file_path):
    """Load MPC inputs from a ROS 2 bag file."""
    storage_id = get_bag_storage_id(bag_file_path)
    storage_options = rosbag2_py.StorageOptions(
        uri=bag_file_path, storage_id=storage_id
    )
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    mpc_inputs = []
    x0s = []
    x0s_timestamp = []
    mpc_inputs_timestamp = []

    while reader.has_next():
        topic, msg, timestamp = reader.read_next()
        if topic == "/mpc_input":
            mpc_inputs.append(
                mpc_msg_to_weighted_traj_point(
                    convert_bytes_to_message(msg, MpcInput), timestamp
                )
            )
            mpc_inputs_timestamp.append(timestamp)
        elif topic == "/ocp_x0":
            x0s.append(sensor_msg_to_numpy(convert_bytes_to_message(msg, Sensor)))
            x0s_timestamp.append(timestamp)

    mpc_inputs_data = {
        "mpc_inputs": mpc_inputs,
        "x0s": x0s,
        "x0s_timestamp": x0s_timestamp,
        "mpc_inputs_timestamp": mpc_inputs_timestamp,
    }
    return mpc_inputs_data


# Define function to convert ROS 2 messages to a pickle file
def save_rosbag_inputs_to_pickle(bag_file_path, pickle_file_path):
    # Open the pickle file
    with open(pickle_file_path, "wb") as pickle_file:
        # Serialize data to pickle file
        pickle.dump(load_mpc_inputs_from_rosbag(bag_file_path), pickle_file)


def load_mpc_outputs_from_rosbag(bag_file_path):
    """Load MPC outputs from a ROS 2 bag file."""
    # Open the rosbag
    storage_id = get_bag_storage_id(bag_file_path)
    storage_options = rosbag2_py.StorageOptions(
        uri=bag_file_path, storage_id=storage_id
    )
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    mpc_data = {}
    mpc_data["states_predictions"] = []
    mpc_data["states_predictions_timestamp"] = []
    mpc_data["control_predictions"] = []
    mpc_data["control_predictions_timestamp"] = []
    mpc_data["kkt_norms"] = []
    mpc_data["kkt_norms_timestamp"] = []
    mpc_data["nb_iters"] = []
    mpc_data["nb_iters_timestamp"] = []
    mpc_data["nb_qp_iters"] = []
    mpc_data["nb_qp_iters_timestamp"] = []
    mpc_data["trajectory_point_id"] = []
    mpc_data["trajectory_point_id_timestamp"] = []
    mpc_data["solve_time"] = []
    mpc_data["solve_time_timestamp"] = []
    mpc_data["pose_detection"] = []
    mpc_data["pose_detection_timestamp"] = []
    mpc_data["mpc_inputs"] = []
    mpc_data["mpc_inputs_timestamp"] = []

    # Read each message in the rosbag and store it
    while reader.has_next():
        topic, msg, timestamp = reader.read_next()
        if topic == "/mpc_debug":
            mpc_debug_msg = convert_bytes_to_message(msg, MpcDebug)
            mpc_data["states_predictions"].append(
                matrix_msg_to_numpy(mpc_debug_msg.states_predictions)
            )
            mpc_data["states_predictions_timestamp"].append(timestamp)
            mpc_data["control_predictions"].append(
                matrix_msg_to_numpy(mpc_debug_msg.control_predictions)
            )
            mpc_data["control_predictions_timestamp"].append(timestamp)
            for residual in mpc_debug_msg.residuals:
                key = residual.name + "_residuals"
                if key not in mpc_data.keys():
                    mpc_data[key] = []
                    mpc_data[key + "_timestamp"] = []
                mpc_data[key].append(matrix_msg_to_numpy(residual.data))
                mpc_data[key + "_timestamp"].append(timestamp)
            for reference in mpc_debug_msg.references:
                key = reference.name + "_references"
                if key not in mpc_data.keys():
                    mpc_data[key] = []
                    mpc_data[key + "_timestamp"] = []
                mpc_data[key].append(matrix_msg_to_numpy(reference.data))
                mpc_data[key + "_timestamp"].append(timestamp)
            mpc_data["kkt_norms"].append(mpc_debug_msg.kkt_norm)
            mpc_data["kkt_norms_timestamp"].append(timestamp)
            mpc_data["nb_iters"].append(mpc_debug_msg.nb_iter)
            mpc_data["nb_iters_timestamp"].append(timestamp)
            mpc_data["nb_qp_iters"].append(mpc_debug_msg.nb_qp_iter)
            mpc_data["nb_qp_iters_timestamp"].append(timestamp)
            trajectory_point_id = 0
            if hasattr(mpc_debug_msg, "trajectory_point_id"):
                trajectory_point_id = mpc_debug_msg.trajectory_point_id
            mpc_data["trajectory_point_id"].append(trajectory_point_id)
            mpc_data["trajectory_point_id_timestamp"].append(timestamp)
        elif topic == "/ocp_solve_time":
            solve_time_msg = convert_bytes_to_message(msg, Duration)
            mpc_data["solve_time"].append(
                solve_time_msg.sec + 1e-9 * solve_time_msg.nanosec
            )
            mpc_data["solve_time_timestamp"].append(timestamp)
        elif topic == "/object/detections":
            pose_msg = convert_bytes_to_message(msg, PoseStamped)
            mpc_data["pose_detection"].append(pose_msg.pose)
            mpc_data["pose_detection_timestamp"].append(timestamp)
        elif topic == "/mpc_input":
            mpc_data["mpc_inputs"].append(
                mpc_msg_to_weighted_traj_point(
                    convert_bytes_to_message(msg, MpcInput), timestamp
                )
            )
            mpc_data["mpc_inputs_timestamp"].append(timestamp)
    return mpc_data


# Define function to convert ROS 2 messages to a pickle file
def save_rosbag_outputs_to_pickle(bag_file_path, pickle_file_path):
    # Open the pickle file
    with open(pickle_file_path, "wb") as pickle_file:
        # Serialize data to pickle file
        pickle.dump(load_mpc_outputs_from_rosbag(bag_file_path), pickle_file)


# Example usage
if __name__ == "__main__":
    bag_file = "/home/gepetto/ros2_ws/src/agimus_controller/bag_files/slow_sin"  # Path to the bag file without the .db3 extension
    pickle_file = "slow_sim_weighted_trajectory_data.pkl"  # Path to the pickle file

    save_rosbag_outputs_to_pickle(bag_file, pickle_file)
    print(f"Saved ROS 2 bag data to {pickle_file}")
