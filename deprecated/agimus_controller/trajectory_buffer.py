from __future__ import annotations
import numpy as np
from collections import deque

from agimus_controller.trajectory_point import TrajectoryPoint, PointAttribute


class TrajectoryBuffer:
    """List of variable size in which the HPP trajectory nodes will be."""

    def __init__(self):
        self._buffer: deque[TrajectoryPoint] = deque()
        print("type ", type(self._buffer))

    def add_trajectory_point(self, trajectory_point: TrajectoryPoint):
        """Add trajectory point to the buffer if it matches the size of q and v"""
        self._buffer.append(trajectory_point)

    def get_size(self, attributes: list[PointAttribute]):
        """Returns the size of the buffer until the first invalid TrajectoryPoint"""
        for idx in range(len(self._buffer) - 1, 0, -1):
            attribute_is_valid = True
            for attribute in attributes:
                if not self._buffer[idx].attribute_is_valid(attribute):
                    attribute_is_valid = False
                    print(
                        f"buffer point at index {idx} is not valid for attribute {attribute}"
                    )
            if attribute_is_valid:
                return idx
        return 0

    def get_points(self, nb_points: int, attributes: list[PointAttribute]):
        """Get nb_points of valid TrajectoryPoints from the buffer"""
        buffer_size = self.get_size(attributes)
        if nb_points > buffer_size:
            raise Exception(
                f"the buffer size is {buffer_size} and you ask for {nb_points} points"
            )
        else:
            return [self._buffer.popleft() for _ in range(nb_points)]

    def get_last_point(self, attributes: list[PointAttribute]):
        """Return last point in buffer without removing it from buffer"""
        buffer_size = self.get_size(attributes)
        if buffer_size == 0:
            return None
        else:
            return self._buffer[-1]

    def get_state_horizon_planning(self):
        """Return the state planning for the horizon, state is composed of joints positions and velocities"""
        nx = len(self._buffer[0].q) + len(self._buffer[0].v)
        x_plan = np.zeros([len(self._buffer), nx])
        for idx, point in enumerate(self._buffer):
            x_plan[idx, :] = np.concatenate([point.q, point.v])
        return x_plan

    def get_joint_acceleration_horizon(self):
        """Return the acceleration reference for the horizon, state is composed of joints positions and velocities"""
        a_plan = np.zeros([len(self._buffer), self.nv])
        for idx, point in enumerate(self._buffer):
            a_plan[idx, :] = point.a
        return a_plan

    def get_buffer(self):
        return self._buffer.copy()
