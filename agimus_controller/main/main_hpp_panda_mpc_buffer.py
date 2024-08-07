import numpy as np

from agimus_controller.hpp_interface import HppInterface
from agimus_controller.mpc import MPC
from agimus_controller.utils.plots import MPCPlots
from agimus_controller.utils.build_models import RobotModelConstructor
from agimus_controller.utils.pin_utils import get_ee_pose_from_configuration
from agimus_controller.utils.wrapper_panda import PandaWrapper
from agimus_controller.ocps.ocp_croco_hpp import OCPCrocoHPP
from agimus_controller.trajectory_buffer import TrajectoryBuffer
from agimus_controller.trajectory_point import TrajectoryPoint, PointAttribute

if __name__ == "__main__":
    pandawrapper = PandaWrapper(auto_col=True)

    robot_constructor = RobotModelConstructor(load_from_ros=False)

    rmodel = robot_constructor.get_robot_reduced_model()
    cmodel = robot_constructor.get_collision_reduced_model()
    vmodel = robot_constructor.get_visual_reduced_model()

    ee_frame_name = pandawrapper.get_ee_frame_name()
    hpp_interface = HppInterface()
    q_init, q_goal = hpp_interface.get_panda_q_init_q_goal()
    hpp_interface.set_panda_planning(q_init, q_goal)
    ps, viewer = hpp_interface.get_problem_solver_and_viewer()
    whole_x_plan, whole_a_plan, whole_traj_T = hpp_interface.get_hpp_x_a_planning(
        1e-2, 7, ps.client.problem.getPath(ps.numberPaths() - 1)
    )
    armature = np.zeros(rmodel.nq)
    ocp = OCPCrocoHPP(rmodel, cmodel, use_constraints=False, armature=armature)
    mpc = MPC(ocp, whole_x_plan, whole_a_plan, rmodel, cmodel)
    mpc.ocp.set_weights(10**4, 1, 10**-3, 0)

    nq = rmodel.nq
    nv = rmodel.nv
    point_attributes = [PointAttribute.Q, PointAttribute.V, PointAttribute.A]
    traj_buffer = TrajectoryBuffer()
    first_point = TrajectoryPoint(nq=nq, nv=nv)
    first_point.q = whole_x_plan[0, :nq]
    first_point.v = whole_x_plan[0, nq:]
    first_point.a = whole_a_plan[0, :]
    traj_buffer.add_trajectory_point(first_point)
    whole_traj_T = whole_x_plan.shape[0]
    T = 100
    mpc_xs = np.zeros([whole_traj_T, 2 * nq])
    mpc_us = np.zeros([whole_traj_T - 1, nq])
    x = first_point.get_x_as_q_v()
    mpc_xs[0, :] = x
    first_step_done = False
    for idx in range(1, whole_traj_T + 2 * T):
        traj_idx = min(idx, whole_x_plan.shape[0] - 1)
        point = TrajectoryPoint(nq=nq, nv=nv)
        point.q = whole_x_plan[traj_idx, :nq]
        point.v = whole_x_plan[traj_idx, nq:]
        point.a = whole_a_plan[traj_idx, :]
        traj_buffer.add_trajectory_point(point)
        if not first_step_done:
            buffer_size = traj_buffer.get_size(point_attributes)
            if buffer_size < 2 * T:
                print(f"buffer size is {buffer_size}, waiting for {2*T} points.")
            else:
                horizon_points = traj_buffer.get_points(T, point_attributes)
                x_plan = np.zeros([T, mpc.nx])
                a_plan = np.zeros([T, mpc.nv])
                for idx_point, point in enumerate(horizon_points):
                    x_plan[idx_point, :] = point.get_x_as_q_v()
                    a_plan[idx_point, :] = point.a
                next_node_idx = T
                x, u = mpc.mpc_first_step(x_plan, a_plan, x, T)
                mpc_xs[idx - 2 * T, :] = x
                mpc_us[idx - 2 * T - 1, :] = u
                first_step_done = True
        else:
            point = traj_buffer.get_points(1, point_attributes)[0]
            new_x_ref = point.get_x_as_q_v()
            new_a_ref = point.a
            placement_ref = get_ee_pose_from_configuration(
                mpc.ocp._rmodel,
                mpc.ocp._rdata,
                mpc.ocp._last_joint_frame_id,
                new_x_ref[:nq],
            )
            x, u = mpc.mpc_step(x, new_x_ref, new_a_ref, placement_ref)
            mpc_xs[idx - 2 * T, :] = x
            mpc_us[idx - 2 * T - 1, :] = u

    u_plan = mpc.ocp.get_u_plan(whole_x_plan, whole_a_plan)
    mpc_plots = MPCPlots(
        croco_xs=mpc_xs,
        croco_us=mpc_us,
        whole_x_plan=whole_x_plan,
        whole_u_plan=u_plan,
        rmodel=rmodel,
        DT=mpc.ocp.DT,
        ee_frame_name=ee_frame_name,
        viewer=viewer,
    )
    mpc_plots.plot_traj()
