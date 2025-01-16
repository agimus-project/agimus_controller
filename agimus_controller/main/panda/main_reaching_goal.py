import time
import numpy as np
from agimus_controller.hpp_interface import HppInterface
from agimus_controller.mpc import MPC
from agimus_controller.utils.path_finder import get_mpc_params_dict
from agimus_controller.visualization.plots import MPCPlots
from agimus_controller.ocps.ocp_pose_ref import OCPPoseRef
from agimus_controller.robot_model.panda_model import (
    get_task_models,
    get_robot_constructor,
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

        rmodel, cmodel, vmodel = get_task_models(task_name="reaching_goal")
        self.robot_constructor = get_robot_constructor(task_name="reaching_goal")
        mpc_params_dict = get_mpc_params_dict(task_name="reaching_goal")
        ocp_params = OCPParameters()
        ocp_params.set_parameters_from_dict(mpc_params_dict["ocp"])

        hpp_interface = HppInterface()
        q_init, q_goal = hpp_interface.get_panda_q_init_q_goal()
        hpp_interface.set_panda_planning(q_init, q_goal, use_gepetto_gui=use_gui)
        viewer = hpp_interface.get_viewer()
        x0 = q_init + [0] * 7
        length = 500
        x_plan = np.array(x0 * length)
        x_plan = np.reshape(x_plan, (length, 14))
        a_plan = np.zeros((length, 7))
        ocp = OCPPoseRef(rmodel, cmodel, ocp_params, np.array(q_goal))

        self.mpc = MPC(ocp, x_plan, a_plan, rmodel, cmodel)
        start = time.time()
        self.mpc.simulate_mpc(save_predictions=True)
        solver = self.mpc.ocp.solver
        costs = return_cost_vectors(solver, weighted=True)
        #constraint = return_constraint_vector(solver)
        plot_costs_from_dic(costs)
        #plot_constraints_from_dic(constraint)
        max_kkt = max(self.mpc.mpc_data["kkt_norm"])
        mean_kkt = np.mean(self.mpc.mpc_data["kkt_norm"])
        mean_iter = np.mean(self.mpc.mpc_data["nb_iter"])
        mean_solve_time = np.mean(self.mpc.mpc_data["solve_time"])
        index = self.mpc.mpc_data["kkt_norm"].index(max_kkt)
        print(f"max kkt {max_kkt} index {index}")
        print(f"mean kkt {mean_kkt} mean iter {mean_iter}")
        print(f"mean solve time {mean_solve_time}")
        end = time.time()
        print("Time of solving: ", end - start)
        ee_frame_name = ocp_params.effector_frame_name
        self.mpc_plots = MPCPlots(
            croco_xs=self.mpc.croco_xs,
            croco_us=self.mpc.croco_us,
            whole_x_plan=x_plan,
            whole_u_plan=np.zeros((length - 1, 7)),
            rmodel=rmodel,
            vmodel=vmodel,
            cmodel=cmodel,
            DT=self.mpc.ocp.params.dt,
            ee_frame_name=ee_frame_name,
            viewer=viewer,
        )

        if use_gui:
            self.mpc_plots.display_path_gepetto_gui()
        return True


def main():
    return APP().main(use_gui=False, spawn_servers=False)


if __name__ == "__main__":
    app = APP()
    app.main(use_gui=True, spawn_servers=True)
