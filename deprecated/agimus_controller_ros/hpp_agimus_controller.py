#!/usr/bin/env python3
import rospy
from agimus_controller.trajectory_point import TrajectoryPoint
from agimus_controller.hpp_interface import HppInterface
from agimus_controller_ros.controller_base import (
    ControllerBase,
    AgimusControllerNodeParameters,
)


class HppAgimusController(ControllerBase):
    def __init__(self, params: AgimusControllerNodeParameters) -> None:
        super().__init__(params)

        self.q_goal = [-0.8311, 0.6782, 0.3201, -1.1128, 1.2190, 1.9823, 0.7248]
        self.hpp_interface = HppInterface()
        self.plan_is_set = False
        self.traj_idx = 0

    def update_state_machine(self):
        return

    def get_next_trajectory_point(self):
        if not self.plan_is_set:
            self.set_plan()
            self.plan_is_set = True

        if self.traj_idx >= self.whole_x_plan.shape[0] - 1:
            return None
        point = TrajectoryPoint(nq=self.nq, nv=self.nv)
        point.q = self.whole_x_plan[self.traj_idx, : self.nq]
        point.v = self.whole_x_plan[self.traj_idx, self.nq :]
        point.a = self.whole_a_plan[self.traj_idx, :]

        self.traj_idx += 1
        return point

    def set_plan(self):
        sensor_msg = self.get_sensor_msg()
        q_init = self.get_x0_from_sensor_msg(sensor_msg)[: self.nq]
        self.hpp_interface.set_panda_planning(
            q_init, self.q_goal, use_gepetto_gui=False
        )
        (
            self.whole_x_plan,
            self.whole_a_plan,
            _,
        ) = self.hpp_interface.get_hpp_x_a_planning(self.params.dt)


def crocco_motion_server_node():
    rospy.init_node("croccodyl_motion_server_node_py", anonymous=True)
    node = HppAgimusController()
    node.run()


if __name__ == "__main__":
    try:
        crocco_motion_server_node()
    except rospy.ROSInterruptException:
        pass
