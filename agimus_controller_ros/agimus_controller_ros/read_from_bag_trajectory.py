from rclpy.serialization import deserialize_message
import rosbag2_py
import pickle
from builtin_interfaces.msg import Duration
from linear_feedback_controller_msgs_py.numpy_conversions import matrix_msg_to_numpy
from agimus_controller_ros.ros_utils import mpc_msg_to_weighted_traj_point
from agimus_msgs.msg import MpcInput, MpcDebug
from geometry_msgs.msg import PoseStamped
from linear_feedback_controller_msgs.msg import Sensor
from linear_feedback_controller_msgs_py.numpy_conversions import sensor_msg_to_numpy


def convert_bytes_to_message(serialized_bytes, msg_type):
    """Deserialize the bytes into the correct message type."""
    message = deserialize_message(serialized_bytes, msg_type)
    return message


def load_mpc_inputs_from_rosbag(bag_file_path):
    """Load MPC inputs from a ROS 2 bag file."""
    storage_options = rosbag2_py.StorageOptions(uri=bag_file_path, storage_id="sqlite3")
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
    storage_options = rosbag2_py.StorageOptions(uri=bag_file_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    mpc_data = {}
    mpc_data["states_predictions"] = []
    mpc_data["control_predictions"] = []
    mpc_data["kkt_norms"] = []
    mpc_data["nb_iters"] = []
    mpc_data["nb_qp_iters"] = []
    mpc_data["trajectory_point_id"] = []
    mpc_data["solve_time"] = []
    mpc_data["pose_detection"] = []
    mpc_data["mpc_inputs"] = []

    # Read each message in the rosbag and store it
    while reader.has_next():
        topic, msg, timestamp = reader.read_next()
        if topic == "/mpc_debug":
            mpc_debug_msg = convert_bytes_to_message(msg, MpcDebug)
            mpc_data["states_predictions"].append(
                matrix_msg_to_numpy(mpc_debug_msg.states_predictions)
            )
            mpc_data["control_predictions"].append(
                matrix_msg_to_numpy(mpc_debug_msg.control_predictions)
            )
            for residual in mpc_debug_msg.residuals:
                if residual.name + "_residuals" not in mpc_data.keys():
                    mpc_data[residual.name + "_residuals"] = []
                mpc_data[residual.name + "_residuals"].append(
                    matrix_msg_to_numpy(residual.data)
                )
            for reference in mpc_debug_msg.references:
                if reference.name + "_references" not in mpc_data.keys():
                    mpc_data[reference.name + "_references"] = []
                mpc_data[reference.name + "_references"].append(
                    matrix_msg_to_numpy(reference.data)
                )
            mpc_data["kkt_norms"].append(mpc_debug_msg.kkt_norm)
            mpc_data["nb_iters"].append(mpc_debug_msg.nb_iter)
            mpc_data["nb_qp_iters"].append(mpc_debug_msg.nb_qp_iter)
            trajectory_point_id = 0
            if hasattr(mpc_debug_msg, "trajectory_point_id"):
                trajectory_point_id = mpc_debug_msg.trajectory_point_id
            mpc_data["trajectory_point_id"].append(trajectory_point_id)
        elif topic == "/ocp_solve_time":
            solve_time_msg = convert_bytes_to_message(msg, Duration)
            mpc_data["solve_time"].append(
                solve_time_msg.sec + 1e-9 * solve_time_msg.nanosec
            )
        elif topic == "/object/detections":
            pose_msg = convert_bytes_to_message(msg, PoseStamped)
            mpc_data["pose_detection"].append(pose_msg.pose)
        elif topic == "/mpc_input":
            mpc_data["mpc_inputs"].append(
                mpc_msg_to_weighted_traj_point(
                    convert_bytes_to_message(msg, MpcInput), timestamp
                )
            )
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
