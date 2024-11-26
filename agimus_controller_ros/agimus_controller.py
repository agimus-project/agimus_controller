import time
from agimus_controller_ros.controller_base import (
    ControllerBase,
    AgimusControllerNodeParameters,
)

from agimus_controller_ros.hpp_subscriber import HPPSubscriber


class AgimusControllerNode(ControllerBase):
    def __init__(self, params: AgimusControllerNodeParameters) -> None:
        super().__init__(params)
        self.hpp_subscriber = HPPSubscriber(use_ros_params=params.use_ros_params)

    def get_next_trajectory_point(self):
        return self.hpp_subscriber.get_trajectory_point()

    def update_state_machine(self):
        if self.hpp_subscriber.last_time_got_traj_point is None:
            return
        self.last_elapsed_time = self.elapsed_time
        self.elapsed_time = time.time() - self.hpp_subscriber.last_time_got_traj_point
        if self.last_elapsed_time is None:
            if self.elapsed_time is not None:
                self.change_state()
            return
        if (self.elapsed_time < 0.1 and self.last_elapsed_time >= 0.1) or (
            self.elapsed_time >= 0.1 and self.last_elapsed_time < 0.1
        ):
            self.change_state()
