#!/usr/bin/env python
import time
from agimus_controller.robot_model.ur3_model import UR3RobotModel
from agimus_controller_ros.parameters import OCPParameters
from agimus_controller.utils.path_finder import get_mpc_params_dict
from agimus_controller.ocps.ocp_croco_hpp import OCPCrocoHPP
from agimus_controller.mpc import MPC
from agimus_controller.hpp_interface import HppInterface
from agimus_controller.visualization.plots import MPCPlots
from agimus_controller.main.servers import Servers


class APP(object):
    def main(self, use_gui=False, spawn_servers=False):
        if spawn_servers:
            self.servers = Servers()
            self.servers.spawn_servers(use_gui)

        ur3_wrapper = UR3RobotModel.load_model()
        rmodel = ur3_wrapper.get_reduced_robot_model()
        hpp_interface = HppInterface()
        hpp_interface.set_ur3_problem_solver(ur3_wrapper)
        x_plan, a_plan, _ = hpp_interface.get_hpp_x_a_planning(1e-2)
        viewer = hpp_interface.get_viewer()
        mpc_params_dict = get_mpc_params_dict()
        ocp_params = OCPParameters(
            use_ros_params=False, params_dict=mpc_params_dict["ocp"]
        )
        ocp = OCPCrocoHPP(rmodel=rmodel, cmodel=None, params=ocp_params)
        mpc = MPC(ocp, x_plan, a_plan, rmodel)
        start = time.time()
        mpc.simulate_mpc(save_predictions=False)
        end = time.time()
        u_plan = mpc.ocp.get_u_plan(x_plan, a_plan)
        self.mpc_plots = MPCPlots(
            croco_xs=mpc.croco_xs,
            croco_us=mpc.croco_us,
            whole_x_plan=x_plan,
            whole_u_plan=u_plan,
            rmodel=rmodel,
            vmodel=ur3_wrapper.get_reduced_visual_model(),
            cmodel=ur3_wrapper.get_reduced_collision_model(),
            DT=mpc.ocp.params.dt,
            ee_frame_name="wrist_3_joint",
            viewer=viewer,
        )
        print("mpc duration ", end - start)
        return True


def main():
    return APP().main(use_gui=False, spawn_servers=True)


if __name__ == "__main__":
    app = APP()
    app.main(use_gui=True, spawn_servers=True)
