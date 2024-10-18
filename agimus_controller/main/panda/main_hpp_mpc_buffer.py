import numpy as np
from agimus_controller.hpp_interface import HppInterface
from agimus_controller.mpc import MPC
from agimus_controller.utils.path_finder import get_package_path, get_mpc_params_dict
from agimus_controller.visualization.plots import MPCPlots
from agimus_controller.utils.pin_utils import get_ee_pose_from_configuration
from agimus_controller.ocps.ocp_croco_hpp import OCPCrocoHPP
from agimus_controller.trajectory_buffer import TrajectoryBuffer
from agimus_controller.trajectory_point import TrajectoryPoint, PointAttribute
from agimus_controller.robot_model.panda_model import (
    PandaRobotModel,
    PandaRobotModelParameters,
)
from agimus_controller_ros.parameters import OCPParameters
from agimus_controller.main.servers import Servers


class APP(object):
    def main(self, use_gui=False, spawn_servers=False):
        if spawn_servers:
            self.servers = Servers()
            self.servers.spawn_servers(use_gui)

        panda_params = PandaRobotModelParameters()
        panda_params.collision_as_capsule = True
        panda_params.self_collision = False
        agimus_demos_description_dir = get_package_path("agimus_demos_description")
        collision_file_path = (
            agimus_demos_description_dir / "pick_and_place" / "obstacle_params.yaml"
        )
        pandawrapper = PandaRobotModel.load_model(
            params=panda_params, env=collision_file_path
        )

        rmodel = pandawrapper.get_reduced_robot_model()
        cmodel = pandawrapper.get_reduced_collision_model()
        ee_frame_name = panda_params.ee_frame_name
        mpc_params_dict = get_mpc_params_dict()
        ocp_params = OCPParameters(
            use_ros_params=False, params_dict=mpc_params_dict["ocp"]
        )

        hpp_interface = HppInterface()
        q_init, q_goal = hpp_interface.get_panda_q_init_q_goal()
        hpp_interface.set_panda_planning(q_init, q_goal, use_gepetto_gui=use_gui)
        viewer = hpp_interface.get_viewer()
        whole_x_plan, whole_a_plan, whole_traj_T = hpp_interface.get_hpp_x_a_planning(
            1e-2
        )
        ocp = OCPCrocoHPP(rmodel, cmodel, ocp_params)
        mpc = MPC(ocp, whole_x_plan, whole_a_plan, rmodel, cmodel)

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
        T = ocp_params.horizon_size
        mpc_xs = np.zeros([whole_traj_T, 2 * nq])
        mpc_us = np.zeros([whole_traj_T - 1, nq])
        x = first_point.get_x_as_q_v()
        mpc_xs[0, :] = x
        first_step_done = False
        for idx in range(1, whole_traj_T + 2 * T):
            traj_idx = min(idx, whole_traj_T - 1)
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
                    # next_node_idx = T
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
                    mpc.ocp._effector_frame_id,
                    new_x_ref[:nq],
                )
                x, u = mpc.mpc_step(x, new_x_ref, new_a_ref, placement_ref)
                mpc_xs[idx - 2 * T, :] = x
                mpc_us[idx - 2 * T - 1, :] = u

        u_plan = mpc.ocp.get_u_plan(whole_x_plan, whole_a_plan)
        self.mpc_plots = MPCPlots(
            croco_xs=mpc_xs,
            croco_us=mpc_us,
            whole_x_plan=whole_x_plan,
            whole_u_plan=u_plan,
            rmodel=rmodel,
            vmodel=pandawrapper.get_reduced_visual_model(),
            cmodel=cmodel,
            DT=mpc.ocp.params.dt,
            ee_frame_name=ee_frame_name,
            viewer=viewer,
        )
        if use_gui:
            self.mpc_plots.display_path_gepetto_gui()
            self.mpc_plots.plot_traj()
        return True


def main():
    return APP().main(use_gui=False, spawn_servers=False)


if __name__ == "__main__":
    app = APP()
    app.main(use_gui=True, spawn_servers=True)
