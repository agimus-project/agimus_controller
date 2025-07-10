import time
import numpy as np
from pathlib import Path
import yaml

from agimus_controller.trajectories.generic_trajectory import GenericTrajectory
import agimus_controller_examples
from agimus_controller_examples.utils.set_models_and_mpc import (
    get_panda_models,
    get_mpc,
)
from agimus_demo_05_pick_and_place.hpp_client import (
    HPPInterface,
    get_q_dq_ddq_arrays_from_path,
)
from agimus_controller.plots.plots_utils import plot_mpc_data
from agimus_controller_examples.utils.wrapper_meshcat import MeshcatWrapper
from ament_index_python.packages import get_package_share_directory


def get_weights(weights, size):
    """
    Return weights with right size if user sent only one value, otherwise
    directly returns weights.
    """
    if len(weights) == 1:
        return np.array(weights * size)
    else:
        return np.array(weights)


class APP(object):
    def __init__(self):
        """Initialize mpc data dictionary."""
        self.mpc_data = {}
        self.mpc_data["states_predictions"] = []
        self.mpc_data["control_predictions"] = []
        self.mpc_data["kkt_norms"] = []
        self.mpc_data["nb_iters"] = []
        self.mpc_data["nb_qp_iters"] = []
        self.mpc_data["trajectory_point_id"] = []
        self.mpc_data["solve_time"] = []

    def fill_mpc_data(self, solve_time: float) -> None:
        """Fill mpc data dictionary."""
        self.mpc_data["solve_time"].append(solve_time)
        self.mpc_data["states_predictions"].append(
            np.array(self.mpc.mpc_debug_data.ocp.result.states)
        )
        self.mpc_data["control_predictions"].append(
            np.array(self.mpc.mpc_debug_data.ocp.result.feed_forward_terms)
        )
        self.mpc_data["kkt_norms"].append(
            np.array(self.mpc.mpc_debug_data.ocp.kkt_norm)
        )
        self.mpc_data["nb_iters"].append(np.array(self.mpc.mpc_debug_data.ocp.nb_iter))
        self.mpc_data["nb_qp_iters"].append(
            np.array(self.mpc.mpc_debug_data.ocp.nb_qp_iter)
        )
        self.mpc_data["trajectory_point_id"].append(
            self.mpc.mpc_debug_data.reference_id
        )
        for name, data in self.mpc.mpc_debug_data.ocp.references:
            if name + "_references" not in self.mpc_data.keys():
                self.mpc_data[name + "_references"] = []
            self.mpc_data[name + "_references"].append(np.asarray(data.copy()))

    def display_path_meshcat(self, xs):
        """Display in Meshcat the trajectory found with crocoddyl."""
        for x in xs:
            self.vis[0].display(x[:7])
            time.sleep(0.01)

    def main(self):
        # set start and goal poses for pick and place task
        start_obj_pose = (
            "panda/support_link",
            [0.1, -0.2, 1.0, 0.0, 0.0, 0.707, 0.707],
        )
        goal_obj_pose = ("dest_box/base_link", [0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0])
        q_init = [
            -0.3619834760502907,
            -1.3575006398318104,
            0.969610481368033,
            -2.6028532848927295,
            0.2040785081450368,
            1.9436352693107668,
            0.6423896937386857,
            0.0,
            0.0,
        ]

        # get mpc and trajectory params
        config_folder_path = (
            Path(agimus_controller_examples.__path__[0])
            / "main"
            / "panda_pick_and_place"
            / "config"
        )
        env_xacro_path = (
            Path(get_package_share_directory("agimus_demo_05_pick_and_place"))
            / "urdf"
            / "environment.urdf.xacro"
        )
        robot_models = get_panda_models(config_folder_path, env_xacro_path)
        self.mpc = get_mpc(config_folder_path)
        with open(config_folder_path / "trajectory_weigths_params.yaml", "r") as file:
            traj_params = yaml.safe_load(file)["simple_trajectory_publisher"][
                "ros__parameters"
            ]
        with open(config_folder_path / "agimus_controller_params.yaml", "r") as file:
            mpc_params = yaml.safe_load(file)["agimus_controller_node"][
                "ros__parameters"
            ]
            dt = mpc_params["ocp"]["dt"]
        nq = robot_models.robot_model.nq
        self.gen_traj = GenericTrajectory(
            traj_params["ee_frame_name"],
            get_weights(traj_params["w_q"], nq),
            get_weights(traj_params["w_qdot"], nq),
            get_weights(traj_params["w_qddot"], nq),
            get_weights(traj_params["w_robot_effort"], nq),
            get_weights(traj_params["w_pose"], 6),
            traj_params["w_collision_avoidance"],
        )
        self.gen_traj.initialize(robot_models.robot_model, q_init)

        # make path planning
        self.hpp_interface = HPPInterface(
            object_name="obj_31",
            use_spline_gradient_based_opt=False,
        )
        self.hpp_interface.set_relative_start_obj_pose(
            start_obj_pose[1], q_init, start_obj_pose[0]
        )
        self.hpp_interface.set_goal_obj_pose(goal_obj_pose[0], goal_obj_pose[1][:3])

        grasp_path, placing_path, freefly_path = self.hpp_interface.plan_pick_and_place(
            q_init=q_init
        )

        # add trajectory points in mpc buffer and make mpc iteration
        t = 0.0
        for path in [grasp_path, placing_path, freefly_path]:
            q_array, dq_array, ddq_array = get_q_dq_ddq_arrays_from_path(path, dt=dt)
            traj = self.gen_traj.build_trajectory_from_q_dq_ddq_arrays(
                q_array, dq_array, ddq_array
            )
            self.gen_traj.add_trajectory(traj)

            for _ in range(len(q_array)):
                w_traj_point = self.gen_traj.get_traj_point_at_t(t)
                t += dt
                self.mpc._buffer.append(w_traj_point)
                self.mpc_debug_data_list = []

                if len(self.mpc._buffer) > self.mpc._buffer.horizon_indexes[-1]:
                    start_solve_time = time.time()
                    x0_traj_point = self.mpc._buffer._buffer[0]
                    self.mpc.run(
                        initial_state=x0_traj_point.point,
                        current_time_ns=x0_traj_point.point.time_ns,
                    )
                    solve_time = time.time() - start_solve_time
                    self.fill_mpc_data(solve_time)
        print("create meshcat objects")
        self.MeshcatVis = MeshcatWrapper()
        print("created wrapper")
        self.vis = self.MeshcatVis.visualize(
            robot_model=robot_models.robot_model,
            robot_collision_model=robot_models.collision_model,
            robot_visual_model=robot_models.visual_model,
        )
        print("finish creating meshcat objects")
        self.xs = np.array(app.mpc_data["states_predictions"])[:, 0, :]

        # plots
        which_plots = [
            "computation_time",
            "iter",
            "predictions",
        ]
        mpc_config = {
            "dt_ocp": dt,
            "mpc_freq": mpc_params["rate"],
            "nb_running_nodes": mpc_params["ocp"]["horizon_size"],
            "endeff_name": traj_params["ee_frame_name"],
        }
        plot_mpc_data(self.mpc_data, mpc_config, robot_models._robot_model, which_plots)

        return True


def main():
    return APP().main()


if __name__ == "__main__":
    app = APP()
    app.main()
