import os
import numpy as np
import yaml
import xacro
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

from agimus_controller.factory.robot_model import RobotModels, RobotModelParameters
from agimus_controller.ocp_param_traj_tracking_collisions import (
    OCPParamsTrajTrackingCollisions,
)
from agimus_controller.ocp_param_base import DTFactorsNSeq
from agimus_controller.ocp.ocp_traj_tracking_collision_avoidance import (
    OCPCrocoTrajTrackCollAvoidance,
)
from agimus_controller.mpc import MPC
from agimus_controller.warm_start_reference import WarmStartReference
from agimus_controller.trajectory import TrajectoryBuffer

params_path = os.path.join(
    get_package_share_directory("agimus_demo_03_mpc_dummy_traj"),
    "config/agimus_controller_params.yaml",
)


def get_panda_models():
    temp_srdf_path = os.path.join(
        get_package_share_directory("franka_description"),
        "robots/fer/fer.srdf",
    )
    robot_xacro_path = os.path.join(
        get_package_share_directory("franka_description"),
        "robots/fer/fer.urdf.xacro",
    )
    robot_urdf = xacro.process_file(
        robot_xacro_path,
        mappings={
            "with_sc": "true",
        },
    )
    env_xacro_path = (
        Path(get_package_share_directory("agimus_demo_03_mpc_dummy_traj"))
        / "urdf"
        / "obstacles.xacro"
    )
    env_urdf = xacro.process_file(env_xacro_path)

    with open(params_path, "r") as file:
        mpc_params = yaml.safe_load(file)["agimus_controller_node"]["ros__parameters"]
    collision_pairs = [
        (
            mpc_params[collision_pair_name]["first"],
            mpc_params[collision_pair_name]["second"],
        )
        for collision_pair_name in mpc_params["collision_pairs_names"]
    ]
    params = RobotModelParameters(
        robot_urdf=robot_urdf.toxml(),
        env_urdf=env_urdf.toxml(),
        srdf=Path(temp_srdf_path),
        free_flyer=False,
        collision_as_capsule=True,
        self_collision=False,
        armature=np.array(mpc_params["ocp"]["armature"]),
        moving_joint_names=[f"fer_joint{i}" for i in range(1, 8)],
        collision_pairs=collision_pairs,
    )
    robot_models = RobotModels(params)
    return robot_models


def get_mpc():
    robot_models = get_panda_models()
    with open(params_path, "r") as file:
        mpc_params = yaml.safe_load(file)["agimus_controller_node"]["ros__parameters"]
    dt_factors_n_seq_dict = mpc_params["ocp"]["dt_factor_n_seq"]

    dt_factors_n_seq = DTFactorsNSeq(
        factors=dt_factors_n_seq_dict["factors"], dts=dt_factors_n_seq_dict["dts"]
    )
    ocp_params = OCPParamsTrajTrackingCollisions(
        dt=mpc_params["ocp"]["dt"],
        collision_safety_margin=mpc_params["ocp"]["collision_safety_margin"],
        activation_distance_threshold=mpc_params["ocp"][
            "activation_distance_threshold"
        ],
        dt_factor_n_seq=dt_factors_n_seq,
        horizon_size=mpc_params["ocp"]["horizon_size"],
        solver_iters=mpc_params["ocp"]["max_iter"],
        callbacks=mpc_params["ocp"]["activate_callback"],
        qp_iters=mpc_params["ocp"]["max_qp_iter"],
    )

    ocp = OCPCrocoTrajTrackCollAvoidance(robot_models, ocp_params)
    ws = WarmStartReference()
    ws.setup(robot_models._robot_model)
    mpc = MPC()
    traj_buffer = TrajectoryBuffer(dt_factors_n_seq)
    mpc.setup(ocp=ocp, warm_start=ws, buffer=traj_buffer)
    return mpc
