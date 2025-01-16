import time
import numpy as np
from agimus_controller.hpp_interface import HppInterface
from agimus_controller.mpc import MPC
from agimus_controller.utils.path_finder import get_package_path, get_mpc_params_dict
from agimus_controller.visualization.plots import MPCPlots
from agimus_controller.ocps.ocp_croco_hpp import OCPCrocoHPP
from agimus_controller.robot_model.panda_model import (
    PandaRobotModel,
    PandaRobotModelParameters,
)
from agimus_controller.ocps.parameters import OCPParameters
from agimus_controller.main.servers import Servers

from agimus_controller.utils.ocp_analyzer import (
    return_cost_vectors,
    return_constraint_vector,
    plot_costs_from_dic,
    plot_constraints_from_dic,
)


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
        mpc_params_dict = get_mpc_params_dict(task_name="pick_and_place")
        ocp_params = OCPParameters()
        ocp_params.set_parameters_from_dict(mpc_params_dict["ocp"])
        rmodel = pandawrapper.get_reduced_robot_model()
        cmodel = pandawrapper.get_reduced_collision_model()
        ee_frame_name = panda_params.ee_frame_name

        self.hpp_interface = HppInterface()
        q_init, q_goal = self.hpp_interface.get_panda_q_init_q_goal()
        self.hpp_interface.set_panda_planning(q_init, q_goal, use_gepetto_gui=use_gui)
        viewer = self.hpp_interface.get_viewer()
        #x_plan, a_plan, _ = hpp_interface.get_hpp_x_a_planning(1e-2)
        hpp_traj_dict = np.load("/home/gepetto/ros_ws/src/agimus_controller/agimus_controller/main/panda/" + "hpp_trajectory.npy",allow_pickle=True).item()
        poses = hpp_traj_dict["poses"]
        vels = hpp_traj_dict["vels"]
        self.x_plan = np.concatenate([poses,vels],axis=1)
        self.a_plan = hpp_traj_dict["accs"]

        ocp = OCPCrocoHPP(rmodel, cmodel, ocp_params)

        self.mpc = MPC(ocp, self.x_plan, self.a_plan, rmodel, cmodel)

        start = time.time()
        self.mpc.simulate_mpc(save_predictions=True)
        end = time.time()
        print("Time of solving: ", end - start)
        max_kkt = max(self.mpc.mpc_data["kkt_norm"])
        mean_kkt = np.mean(self.mpc.mpc_data["kkt_norm"])
        mean_iter = np.mean(self.mpc.mpc_data["nb_iter"])
        mean_solve_time = np.mean(self.mpc.mpc_data["step_time"])
        index = self.mpc.mpc_data["kkt_norm"].index(max_kkt)
        print(f"max kkt {max_kkt} index {index}")
        print(f"mean kkt {mean_kkt} mean iter {mean_iter}")
        print(f"mean solve time {mean_solve_time}")
        costs = return_cost_vectors(self.mpc.ocp.solver, weighted=True)
        constraint = return_constraint_vector(self.mpc.ocp.solver)
        plot_costs_from_dic(costs)
        plot_constraints_from_dic(constraint)
        u_plan = self.mpc.ocp.get_u_plan(self.x_plan, self.a_plan)
        self.mpc_plots = MPCPlots(
            croco_xs=self.mpc.croco_xs,
            croco_us=self.mpc.croco_us,
            whole_x_plan=self.x_plan,
            whole_u_plan=u_plan,
            rmodel=rmodel,
            vmodel=pandawrapper.get_reduced_visual_model(),
            cmodel=cmodel,
            DT=self.mpc.ocp.params.dt,
            ee_frame_name=ee_frame_name,
            viewer=viewer,
        )
        return True


def main():
    return APP().main(use_gui=False, spawn_servers=False)


if __name__ == "__main__":
    app = APP()
    app.main(use_gui=True, spawn_servers=True)
