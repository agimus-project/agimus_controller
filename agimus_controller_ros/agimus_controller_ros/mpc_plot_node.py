from ament_index_python import get_package_share_directory
import numpy as np
from pathlib import Path
import xacro

from agimus_controller.plots.plots_utils import plot_mpc_data
from agimus_controller.factory.robot_model import RobotModelParameters, RobotModels
from agimus_controller_ros.read_from_bag_trajectory import (
    load_mpc_outputs_from_rosbag,
)


def plot_mpc(args) -> None:
    bag_file_path = Path(args.bag_file_path)
    if "panda" in bag_file_path.stem:
        config_folder_path = (
            Path(get_package_share_directory("agimus_demo_03_mpc_dummy_traj"))
            / "config"
        )
        env_xacro_path = (
            Path(get_package_share_directory("agimus_demo_05_pick_and_place"))
            / "urdf"
            / "environment.urdf.xacro"
        )
        robot_models = RobotModels(
            RobotModelParameters.from_panda(config_folder_path, env_xacro_path)
        )
    elif (
        "tiago_pro" in bag_file_path.stem.lower()
        or "tiago-pro" in bag_file_path.stem.lower()
    ):
        import tempfile

        tiago_pro_pkg = Path(get_package_share_directory("tiago_pro_description"))
        tiago_pro_urdf_path = tiago_pro_pkg / "robots" / "tiago_pro.urdf.xacro"
        tiago_pro_urdf = xacro.process_file(tiago_pro_urdf_path).toxml()
        tiago_pro_moveit_pkg = Path(
            get_package_share_directory("tiago_pro_moveit_config")
        )
        tiago_pro_srdf_path = (
            tiago_pro_moveit_pkg / "config" / "srdf" / "tiago_pro.srdf.xacro"
        )
        tiago_pro_srdf_xml = xacro.process_file(
            tiago_pro_srdf_path,
            mappings={
                "end_effector_left": "pal-pro-gripper",
                "end_effector_right": "pal-pro-gripper",
            },
        ).toxml()
        # Write SRDF XML to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".srdf") as tmp_srdf:
            tmp_srdf.write(tiago_pro_srdf_xml.encode("utf-8"))
            tmp_srdf_path = Path(tmp_srdf.name)
        moving_joint_names = [
            "arm_right_1_joint",
            "arm_right_2_joint",
            "arm_right_3_joint",
            "arm_right_4_joint",
            "arm_right_5_joint",
            "arm_right_6_joint",
            "arm_right_7_joint",
        ]
        params = RobotModelParameters(
            robot_urdf=tiago_pro_urdf,
            env_urdf=None,
            srdf=tmp_srdf_path,
            free_flyer=False,
            armature=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
            moving_joint_names=moving_joint_names,
        )
        robot_models = RobotModels(params)
    else:
        raise ValueError(
            "Unsupported robot type in bag file name. "
            "Please use a bag file with 'panda', 'tiago_pro', or 'tiago-pro' in its name."
        )
    mpc_data = load_mpc_outputs_from_rosbag(args.bag_file_path)
    which_plots = [
        "computation_time",
        "collision_distance",
        "iter",
        "visual_servoing",
        "predictions",
    ]
    mpc_config = {
        "dt_ocp": args.dt_ocp,
        "nb_running_nodes": mpc_data["control_predictions"][0].shape[0],
        "endeff_name": args.endeff_name,
        "mpc_freq": 1.0 / args.dt_ocp,
    }

    plot_mpc_data(mpc_data, mpc_config, robot_models.robot_model, which_plots)
    return


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot MPC data from a ROS bag file.")
    parser.add_argument(
        "--bag-file-path",
        "-p",
        type=str,
        required=True,
        help="Path to the ROS bag file containing MPC data.",
    )
    parser.add_argument(
        "--dt_ocp",
        "-t",
        type=float,
        required=True,
        help="Time step for the OCP (Optimal Control Problem) in seconds.",
    )
    parser.add_argument(
        "--endeff_name",
        "-e",
        type=str,
        required=True,
        help="Name of the end effector in the robot model.",
    )
    args = parser.parse_args()
    plot_mpc(args)


if __name__ == "__main__":
    main()
