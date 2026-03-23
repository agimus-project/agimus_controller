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
    delete_tmp_srdf = False
    print(f"Loading data for bag file: {bag_file_path}")
    print("Determining robot type...")
    if args.robot:
        robot = args.robot
    else:
        if "panda" in bag_file_path.stem:
            robot = "panda"
        elif (
            "tiago_pro" in bag_file_path.stem.lower()
            or "tiago-pro" in bag_file_path.stem.lower()
        ):
            robot = "tiago-pro"
        else:
            raise ValueError(
                f"Unsupported robot type in argument {args.robot}. "
                "Please use a bag file with 'panda', 'tiago_pro', or 'tiago-pro' in its name."
            )
    print(f"Robot type determined: {robot}")
    if robot == "panda":
        agimus_franka_pkg = Path(
            get_package_share_directory("agimus_franka_description")
        )
        agimus_franka_urdf_path = (
            agimus_franka_pkg / "robots" / "fer" / "fer.urdf.xacro"
        )
        agimus_franka_urdf = xacro.process_file(agimus_franka_urdf_path).toxml()
        agimus_franka_srdf_path = agimus_franka_pkg / "robots" / "fer" / "fer.srdf"
        moving_joint_names = [
            "fer_joint1",
            "fer_joint2",
            "fer_joint3",
            "fer_joint4",
            "fer_joint5",
            "fer_joint6",
            "fer_joint7",
        ]
        params = RobotModelParameters(
            robot_urdf=agimus_franka_urdf,
            env_urdf=None,
            srdf=agimus_franka_srdf_path,
            free_flyer=False,
            armature=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
            moving_joint_names=moving_joint_names,
        )
    elif robot == "tiago-pro":
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
        tmp_srdf = tempfile.NamedTemporaryFile(delete=False, suffix=".srdf")
        tmp_srdf.write(tiago_pro_srdf_xml.encode("utf-8"))
        tmp_srdf.close()
        tmp_srdf_path = Path(tmp_srdf.name)
        delete_tmp_srdf = True
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
    else:
        raise ValueError(
            "Unsupported robot type in bag file name. "
            "Please use a bag file with 'panda', 'tiago_pro', or 'tiago-pro' in its name."
        )

    robot_models = RobotModels(params)
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

    plot_mpc_data(
        mpc_data,
        mpc_config,
        robot_models.robot_model,
        which_plots,
        dump_path=str(bag_file_path),
    )

    if delete_tmp_srdf:
        tmp_srdf_path.unlink()
    return


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot MPC data from a ROS bag file.")
    parser.add_argument(
        "--robot",
        "-r",
        type=str,
        required=False,
        default=None,
        choices=["panda", "tiago_pro", "tiago-pro"],
        help="Name of the robot.",
    )
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
    parser.add_argument(
        "--save-figures",
        "-s",
        action="store_true",
        default=False,
        help="If set, save the generated figures instead of displaying them.",
    )
    args = parser.parse_args()
    plot_mpc(args)


if __name__ == "__main__":
    main()
