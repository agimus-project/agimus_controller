#!/usr/bin/env python3
import rospy
import numpy as np
from copy import deepcopy
import time
from threading import Lock
from std_msgs.msg import Duration, Header
from linear_feedback_controller_msgs.msg import Control, Sensor
import atexit

from agimus_controller.utils.ros_np_multiarray import to_multiarray_f64
from agimus_controller.trajectory_buffer import TrajectoryBuffer
from agimus_controller.trajectory_point import PointAttribute
from agimus_controller.utils.build_models import RobotModelConstructor
from agimus_controller.utils.pin_utils import (
    get_ee_pose_from_configuration,
    get_last_joint,
)
from agimus_controller.mpc import MPC
from agimus_controller.ocps.ocp_croco_hpp import OCPCrocoHPP


class AgimusControllerNodeParameters:
    def __init__(self) -> None:
        self.rate = rospy.get_param("~rate", 100)
        self.horizon_size = rospy.get_param("~horizon_size", 100)


class ControllerBase:
    def __init__(self) -> None:
        self.dt = 1e-2
        self.params = AgimusControllerNodeParameters()
        self.traj_buffer = TrajectoryBuffer()
        self.point_attributes = [PointAttribute.Q, PointAttribute.V, PointAttribute.A]

        robot_constructor = RobotModelConstructor(load_from_ros=False)

        self.rmodel = robot_constructor.get_robot_reduced_model()
        self.cmodel = robot_constructor.get_collision_reduced_model()

        self.rdata = self.rmodel.createData()
        self.last_joint_name, self.last_joint_id, self.last_joint_frame_id = (
            get_last_joint(self.rmodel)
        )
        self.nq = self.rmodel.nq
        self.nv = self.rmodel.nv
        self.nx = self.nq + self.nv
        self.armature = np.array([0.05] * self.nq)

        self.ocp = OCPCrocoHPP(
            self.rmodel, self.cmodel, use_constraints=False, armature=self.armature
        )
        self.ocp.set_weights(10**4, 10, 10**-3, 0)
        self.save_predictions_and_refs = False
        self.mpc_data = {}

        self.rate = rospy.Rate(self.params.rate, reset=True)
        self.mutex = Lock()
        self.sensor_msg = Sensor()
        self.control_msg = Control()
        self.ocp_solve_time = Duration()
        self.x0 = np.zeros(self.nq + self.nv)
        self.x_guess = np.zeros(self.nq + self.nv)
        self.u_guess = np.zeros(self.nv)
        self.state_subscriber = rospy.Subscriber(
            "robot_sensors",
            Sensor,
            self.sensor_callback,
        )
        self.control_publisher = rospy.Publisher(
            "motion_server_control", Control, queue_size=1, tcp_nodelay=True
        )
        self.ocp_solve_time_pub = rospy.Publisher(
            "ocp_solve_time", Duration, queue_size=1, tcp_nodelay=True
        )
        self.start_time = 0.0
        self.first_robot_sensor_msg_received = False
        self.first_pose_ref_msg_received = True

    def sensor_callback(self, sensor_msg):
        with self.mutex:
            self.sensor_msg = deepcopy(sensor_msg)
            if not self.first_robot_sensor_msg_received:
                self.first_robot_sensor_msg_received = True

    def wait_first_sensor_msg(self):
        wait_for_input = True
        while not rospy.is_shutdown() and wait_for_input:
            wait_for_input = (
                not self.first_robot_sensor_msg_received
                or not self.first_pose_ref_msg_received
            )
            if wait_for_input:
                rospy.loginfo_throttle(3, "Waiting until we receive a sensor message.")
                with self.mutex:
                    sensor_msg = deepcopy(self.sensor_msg)
                    self.start_time = sensor_msg.header.stamp.to_sec()
            rospy.loginfo_once("Start controller")
            self.rate.sleep()
        return wait_for_input

    def wait_buffer_has_twice_horizon_points(self):
        while (
            self.traj_buffer.get_size(self.point_attributes)
            < 2 * self.params.horizon_size
        ):
            self.fill_buffer()

    def get_next_trajectory_point(self):
        raise RuntimeError("Not implemented")

    def fill_buffer(self):
        point = self.get_next_trajectory_point()
        if point is not None:
            self.traj_buffer.add_trajectory_point(point)

    def first_solve(self):
        sensor_msg = self.get_sensor_msg()

        # retrieve horizon state and acc references
        horizon_points = self.traj_buffer.get_points(
            self.params.horizon_size, self.point_attributes
        )
        x_plan = np.zeros([self.params.horizon_size, self.nx])
        a_plan = np.zeros([self.params.horizon_size, self.nv])
        for idx_point, point in enumerate(horizon_points):
            x_plan[idx_point, :] = point.get_x_as_q_v()
            a_plan[idx_point, :] = point.a

        # First solve
        self.mpc = MPC(self.ocp, x_plan, a_plan, self.rmodel, self.cmodel)
        x0 = np.concatenate(
            [sensor_msg.joint_state.position, sensor_msg.joint_state.velocity]
        )
        self.mpc.mpc_first_step(x_plan, a_plan, x0, self.params.horizon_size)
        self.next_node_idx = self.params.horizon_size
        if self.save_predictions_and_refs:
            self.create_mpc_data()
        _, u, k = self.mpc.get_mpc_output()
        return sensor_msg, u, k

    def solve(self):
        sensor_msg = self.get_sensor_msg()
        x0 = np.concatenate(
            [sensor_msg.joint_state.position, sensor_msg.joint_state.velocity]
        )
        self.fill_buffer()
        point = self.traj_buffer.get_points(1, self.point_attributes)[0]
        new_x_ref = point.get_x_as_q_v()
        new_a_ref = point.a

        mpc_start_time = time.time()
        placement_ref = get_ee_pose_from_configuration(
            self.rmodel,
            self.rdata,
            self.last_joint_frame_id,
            new_x_ref[: self.rmodel.nq],
        )
        self.mpc.mpc_step(x0, new_x_ref, new_a_ref, placement_ref)
        mpc_duration = time.time() - mpc_start_time
        rospy.loginfo_throttle(1, "mpc_duration = %s", str(mpc_duration))
        if self.next_node_idx < self.mpc.whole_x_plan.shape[0] - 1:
            self.next_node_idx += 1
        if self.save_predictions_and_refs:
            self.fill_predictions_and_refs_arrays()
        _, u, k = self.mpc.get_mpc_output()

        return sensor_msg, u, k

    def get_sensor_msg(self):
        with self.mutex:
            sensor_msg = deepcopy(self.sensor_msg)
        return sensor_msg

    def send(self, sensor_msg, u, k):
        self.control_msg.header = Header()
        self.control_msg.header.stamp = rospy.Time.now()
        self.control_msg.feedback_gain = to_multiarray_f64(k)
        self.control_msg.feedforward = to_multiarray_f64(u)
        self.control_msg.initial_state = sensor_msg
        self.control_publisher.publish(self.control_msg)

    def create_mpc_data(self):
        xs, us = self.mpc.get_predictions()
        x_ref, p_ref, u_ref = self.mpc.get_reference()
        self.mpc_data["preds_xs"] = xs[np.newaxis, :]
        self.mpc_data["preds_us"] = us[np.newaxis, :]
        self.mpc_data["state_refs"] = x_ref[np.newaxis, :]
        self.mpc_data["translation_refs"] = p_ref[np.newaxis, :]
        self.mpc_data["control_refs"] = u_ref[np.newaxis, :]

    def fill_predictions_and_refs_arrays(self):
        xs, us = self.mpc.get_predictions()
        x_ref, p_ref, u_ref = self.mpc.get_reference()
        self.mpc_data["preds_xs"] = np.r_[self.mpc_data["preds_xs"], xs[np.newaxis, :]]
        self.mpc_data["preds_us"] = np.r_[self.mpc_data["preds_us"], us[np.newaxis, :]]
        self.mpc_data["state_refs"] = np.r_[
            self.mpc_data["state_refs"], x_ref[np.newaxis, :]
        ]
        self.mpc_data["translation_refs"] = np.r_[
            self.mpc_data["translation_refs"], p_ref[np.newaxis, :]
        ]
        self.mpc_data["control_refs"] = np.r_[
            self.mpc_data["control_refs"], u_ref[np.newaxis, :]
        ]

    def exit_handler(self):
        np.save("mpc_data.npy", self.mpc_data)

    def run(self):
        self.wait_first_sensor_msg()
        self.wait_buffer_has_twice_horizon_points()
        sensor_msg, u, k = self.first_solve()
        input("Press enter to continue ...")
        self.send(sensor_msg, u, k)
        self.rate.sleep()
        if self.save_predictions_and_refs:
            atexit.register(self.exit_handler)
        while not rospy.is_shutdown():
            start_compute_time = rospy.Time.now()
            sensor_msg, u, k = self.solve()
            self.send(sensor_msg, u, k)
            self.rate.sleep()
            self.ocp_solve_time.data = rospy.Time.now() - start_compute_time
            self.ocp_solve_time_pub.publish(self.ocp_solve_time)
