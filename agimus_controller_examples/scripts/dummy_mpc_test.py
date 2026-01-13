import example_robot_data as robex
import pinocchio as pin
import numpy as np
from pathlib import Path

from agimus_controller.trajectory import (
    WeightedTrajectoryPoint,
    TrajectoryPoint,
    TrajectoryPointWeights,
)
from agimus_controller.ocp.ocp_croco_generic import OCPCrocoGeneric
from agimus_controller.trajectory import TrajectoryBuffer
from agimus_controller.factory.robot_model import RobotModels, RobotModelParameters
from agimus_controller.ocp_param_base import OCPParamsBaseCroco
from agimus_controller.warm_start_reference import WarmStartReference
from agimus_controller.mpc import MPC
from agimus_controller.ocp_param_base import DTFactorsNSeq
from agimus_controller.factory import ocp_yaml_parser

from pinocchio.visualize import MeshcatVisualizer

if __name__ == "__main__":
    robot = robex.load("panda")

    urdf_path = Path(robot.urdf)
    srdf_path = Path(robot.urdf.replace("urdf", "srdf"))
    urdf_meshes_dir = urdf_path.parent.parent.parent.parent.parent

    robot_params = RobotModelParameters(
        free_flyer=False,
        moving_joint_names=[
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ],
        robot_urdf=urdf_path,
        srdf=srdf_path,
        urdf_meshes_dir=urdf_meshes_dir,
        collision_as_capsule=True,
        self_collision=True,
        armature=np.ones(7) * 0.05,
    )

    robot_models = RobotModels(robot_params)

    vizer = MeshcatVisualizer(
        robot_models.robot_model,
        robot_models.collision_model,
        robot_models.visual_model,
    )
    vizer.initViewer(zmq_url="tcp://127.0.0.1:6000")
    vizer.loadViewerModel()

    horizon_size = 40
    total_steps = 3000
    dt = 0.01
    dt_factor_n_seq = DTFactorsNSeq(factors=[1], n_steps=[horizon_size])
    ocp_params = OCPParamsBaseCroco(
        dt_factor_n_seq=dt_factor_n_seq,
        dt=dt,
        horizon_size=horizon_size,
        solver_iters=20,
        callbacks=True,
        qp_iters=100,
        use_debug_data=False,
        n_threads=8,
    )

    ocp_definition_file = ocp_yaml_parser.get_default_yaml_file(
        "ocp_goal_reaching.yaml"
    )
    ocp = OCPCrocoGeneric(
        robot_models, ocp_params, ocp_definition_file, expect_rolling_buffer=True
    )

    traj_buffer = TrajectoryBuffer(dt_factor_n_seq)

    # Use WarmStartReference for initialization
    ws_ref = WarmStartReference()
    ws_ref.setup(robot_models.robot_model)

    mpc = MPC()
    mpc.setup(ocp=ocp, warm_start=ws_ref, buffer=traj_buffer)

    q0 = np.array([0.0, -0.78, 0.0, -2.35, 0.0, 1.57, 0.78])
    dq0 = np.zeros_like(q0)
    ddq0 = np.zeros_like(q0)

    robot_data = robot_models.robot_model.createData()
    tau0 = pin.rnea(robot_models.robot_model, robot_data, q0, dq0, dq0)

    for i in range(total_steps):
        wpt = WeightedTrajectoryPoint(
            TrajectoryPoint(
                robot_configuration=q0,
                robot_velocity=dq0,
                robot_acceleration=ddq0,
                robot_effort=tau0,
                end_effector_poses={
                    "panda_hand_tcp": pin.SE3(
                        pin.Quaternion(np.array([1.0, 0.0, 0.0, 0.0])),
                        np.array([0.5, np.sin(i * dt) * 0.2, 0.5]),
                    )
                },
            ),
            TrajectoryPointWeights(
                w_robot_configuration=0.3 * np.ones(robot_models.robot_model.nv),
                w_robot_velocity=0.2 * np.ones(robot_models.robot_model.nv),
                w_robot_effort=0.1 * np.ones(robot_models.robot_model.nv),
                w_end_effector_poses={"panda_hand_tcp": 10 * np.ones(6)},
            ),
        )
        mpc.append_trajectory_point(wpt)

    for i in range(total_steps - horizon_size):
        x0_traj_point = TrajectoryPoint(
            time_ns=0,
            robot_configuration=q0,
            robot_velocity=dq0,
            robot_acceleration=ddq0,
            forces=pin.Force.Zero(),
        )
        ocp_res = mpc.run(x0_traj_point, 0)
        q0 = ocp_res.states[1][:7]
        dq0 = ocp_res.states[1][7:]
        vizer.display(q0)
