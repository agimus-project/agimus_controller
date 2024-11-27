#!/usr/bin/env python3
import rospy
import numpy as np
from copy import deepcopy
import time
from threading import Lock
from std_msgs.msg import Duration, Header
from linear_feedback_controller_msgs.msg import Control, Sensor
import atexit

from agimus_controller.hpp_interface import HppInterface
from agimus_controller_ros.ros_np_multiarray import to_multiarray_f64
from agimus_controller.robot_model.panda_model import get_task_models
from agimus_controller.utils.pin_utils import get_ee_pose_from_configuration
from agimus_controller.mpc import MPC
from agimus_controller.ocps.ocp_pose_ref import OCPPoseRef
from agimus_controller_ros.sim_utils import convert_float_to_ros_duration_msg
from agimus_controller_ros.parameters import AgimusControllerNodeParameters


class ReachingGoalController:
    def __init__(self, params: AgimusControllerNodeParameters) -> None:
        self.params = params
        self.rmodel, self.cmodel, self.vmodel = get_task_models(
            task_name="reaching_goal"
        )
        self.rdata = self.rmodel.createData()
        self.effector_frame_id = self.rmodel.getFrameId(
            self.params.ocp.effector_frame_name
        )
        self.hpp_interface = HppInterface()
        self.nq = self.rmodel.nq
        self.nv = self.rmodel.nv
        self.nx = self.nq + self.nv

        q_init, q_goal = self.hpp_interface.get_panda_q_init_q_goal()
        self.ocp = OCPPoseRef(self.rmodel, self.cmodel, params.ocp, np.array(q_goal))

        if self.params.use_ros_params:
            self.initialize_ros_attributes()

    def initialize_ros_attributes(self):
        self.rate = rospy.Rate(self.params.rate, reset=True)
        self.mutex = Lock()
        self.sensor_msg = Sensor()
        self.control_msg = Control()
        self.state_subscriber = rospy.Subscriber(
            "robot_sensors", Sensor, self.sensor_callback
        )
        self.control_publisher = rospy.Publisher(
            "motion_server_control", Control, queue_size=1, tcp_nodelay=True
        )
        self.ocp_solve_time_pub = rospy.Publisher(
            "ocp_solve_time", Duration, queue_size=1, tcp_nodelay=True
        )
        self.ocp_x0_pub = rospy.Publisher(
            "ocp_x0", Sensor, queue_size=1, tcp_nodelay=True
        )
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
            rospy.loginfo_once("Start controller")
            self.rate.sleep()
        return wait_for_input

    def first_solve(self, x0):
        # retrieve horizon state and acc references
        # x0 = list(x0)
        x_plan = np.array(list(x0) * self.params.ocp.horizon_size)
        x_plan = np.reshape(x_plan, (self.params.ocp.horizon_size, 14))
        a_plan = np.zeros((self.params.ocp.horizon_size, 7))

        # x0 = np.array(x0)

        # First solve
        self.mpc = MPC(self.ocp, x_plan, a_plan, self.rmodel, self.cmodel)
        self.mpc.ocp.set_planning_variables(x_plan, a_plan)
        self.mpc.mpc_first_step(
            x_plan, self.ocp.u_plan[: self.params.ocp.horizon_size - 1], x0
        )
        self.next_node_idx = self.params.ocp.horizon_size
        if self.params.save_predictions_and_refs:
            self.mpc.create_mpc_data()
        _, u, k = self.mpc.get_mpc_output()
        return u, k

    def solve(self, x0):
        new_x_ref = np.array(x0)
        new_a_ref = np.zeros((7))
        self.in_world_M_effector = get_ee_pose_from_configuration(
            self.rmodel,
            self.rdata,
            self.effector_frame_id,
            new_x_ref[: self.rmodel.nq],
        )
        # if last point of the pick trajectory is in horizon and we wanna use vision pose
        self.mpc.mpc_step(x0, new_x_ref, new_a_ref, self.in_world_M_effector)

        if self.params.save_predictions_and_refs:
            self.mpc.fill_predictions_and_refs_arrays()
        _, u, k = self.mpc.get_mpc_output()
        return u, k

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

    def exit_handler(self):
        print("saving data")
        np.save("mpc_params.npy", self.params.get_dict())
        if self.params.save_predictions_and_refs:
            np.save("mpc_data.npy", self.mpc.mpc_data)

    def get_x0_from_sensor_msg(self, sensor_msg):
        return np.concatenate(
            [sensor_msg.joint_state.position, sensor_msg.joint_state.velocity]
        )

    def publish_ocp_solve_time(self, ocp_solve_time):
        self.ocp_solve_time_pub.publish(
            convert_float_to_ros_duration_msg(ocp_solve_time)
        )

    def run(self):
        self.wait_first_sensor_msg()
        sensor_msg = self.get_sensor_msg()
        time.sleep(2.0)
        start_compute_time = time.time()
        u, k = self.first_solve(self.get_x0_from_sensor_msg(sensor_msg))
        compute_time = time.time() - start_compute_time
        self.send(sensor_msg, u, k)
        self.publish_ocp_solve_time(compute_time)
        self.ocp_x0_pub.publish(sensor_msg)
        self.rate.sleep()
        atexit.register(self.exit_handler)
        self.run_timer = rospy.Timer(
            rospy.Duration(self.params.ocp.dt), self.run_callback
        )
        rospy.spin()

    def run_callback(self, *args):
        start_compute_time = time.time()
        sensor_msg = self.get_sensor_msg()
        u, k = self.solve(self.get_x0_from_sensor_msg(sensor_msg))
        self.send(sensor_msg, u, k)
        compute_time = time.time() - start_compute_time
        self.publish_ocp_solve_time(compute_time)
        self.ocp_x0_pub.publish(sensor_msg)
