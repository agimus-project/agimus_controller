import yaml
import xacro
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
import numpy as np

from agimus_controller.factory.robot_model import RobotModels, RobotModelParameters
from agimus_controller.ocp_param_base import (
    OCPParamsBaseCroco,
)
from agimus_controller.ocp_param_base import DTFactorsNSeq
from agimus_controller.ocp.ocp_croco_generic import OCPCrocoGeneric
from agimus_controller.mpc import MPC
from agimus_controller.warm_start_reference import WarmStartReference
from agimus_controller.trajectory import TrajectoryBuffer


def get_panda_models(config_folder_path: Path, env_xacro_path=None) -> RobotModels:
    agimus_franka_description_path = Path(
        get_package_share_directory("agimus_franka_description")
    )
    robot_srdf_path = agimus_franka_description_path / "robots" / "fer" / "fer.srdf"
    robot_xacro_path = (
        agimus_franka_description_path / "robots" / "fer" / "fer.urdf.xacro"
    )
    params_path = Path(config_folder_path) / "agimus_controller_params.yaml"
    robot_urdf = xacro.process_file(
        str(robot_xacro_path),
    ).toxml()
    if env_xacro_path is None:
        env_urdf = None
    else:
        env_urdf = xacro.process_file(env_xacro_path).toxml()

    with open(params_path, "r") as file:
        mpc_params = yaml.safe_load(file)["agimus_controller_node"]["ros__parameters"]
    params = RobotModelParameters(
        robot_urdf=robot_urdf,
        env_urdf=env_urdf,
        srdf=Path(robot_srdf_path),
        free_flyer=False,
        armature=np.array(mpc_params["ocp"]["armature"]),
        moving_joint_names=[f"fer_joint{i}" for i in range(1, 8)],
        robot_attachment_frame="robot_attachment_link",
    )
    robot_models = RobotModels(params)
    return robot_models


def get_mpc(config_folder_path: Path) -> MPC:
    robot_models = get_panda_models(config_folder_path)
    params_path = Path(config_folder_path) / "agimus_controller_params.yaml"

    with open(params_path, "r") as file:
        mpc_params = yaml.safe_load(file)["agimus_controller_node"]["ros__parameters"]
    dt_factors_n_seq_dict = mpc_params["ocp"]["dt_factor_n_seq"]
    dt_factors_n_seq = DTFactorsNSeq(
        factors=dt_factors_n_seq_dict["factors"],
        n_steps=dt_factors_n_seq_dict["n_steps"],
    )
    ocp_params = OCPParamsBaseCroco(
        dt=mpc_params["ocp"]["dt"],
        dt_factor_n_seq=dt_factors_n_seq,
        horizon_size=mpc_params["ocp"]["horizon_size"],
        solver_iters=mpc_params["ocp"]["max_iter"],
        callbacks=mpc_params["ocp"]["activate_callback"],
        qp_iters=mpc_params["ocp"]["max_qp_iter"],
        use_debug_data=mpc_params["publish_debug_data"],
    )
    ocp_def_yaml_path = config_folder_path / "ocp_definition_file.yaml"
    ocp = OCPCrocoGeneric(robot_models, ocp_params, ocp_def_yaml_path)

    ws = WarmStartReference()
    ws.setup(robot_models._robot_model)
    mpc = MPC()
    traj_buffer = TrajectoryBuffer(dt_factors_n_seq)
    mpc.setup(ocp=ocp, warm_start=ws, buffer=traj_buffer)
    return mpc
