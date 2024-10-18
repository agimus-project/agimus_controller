#!/usr/bin/env python3
import rospy
import numpy as np
from copy import deepcopy
import time
import pinocchio as pin
from enum import Enum
from threading import Lock
from std_msgs.msg import Duration, Header, Int8
from geometry_msgs.msg import Pose
from vision_msgs.msg import Detection2DArray
import tf2_ros
import tf2_geometry_msgs
from linear_feedback_controller_msgs.msg import Control, Sensor
import atexit

from agimus_controller_ros.ros_np_multiarray import to_multiarray_f64
from agimus_controller.trajectory_buffer import TrajectoryBuffer
from agimus_controller.trajectory_point import PointAttribute
from agimus_controller.robot_model.panda_model import get_pick_and_place_task_models
from agimus_controller.utils.pin_utils import get_ee_pose_from_configuration
from agimus_controller.mpc import MPC
from agimus_controller.ocps.ocp_croco_hpp import OCPCrocoHPP
from agimus_controller_ros.sim_utils import convert_float_to_ros_duration_msg
from agimus_controller_ros.parameters import AgimusControllerNodeParameters


class HPPStateMachine(Enum):
    WAITING_PICK_TRAJECTORY = 1
    RECEIVING_PICK_TRAJECTORY = 2
    WAITING_PLACE_TRAJECTORY = 3
    RECEIVING_PLACE_TRAJECTORY = 4
    WAITING_GOING_INIT_POSE_TRAJECTORY = 5
    RECEIVING_GOING_INIT_POSE_TRAJECTORY = 6


def find_tracked_object(detections):
    tless1_detections = []
    for detection in detections:
        if detection.results[0].hypothesis.class_id != "tless-obj_000001":
            continue
        else:
            tless1_detections.append(detection)
    if len(tless1_detections) == 0:
        return None
    if len(tless1_detections) == 1:
        return tless1_detections[0].results[0].pose
    current_idx = int(tless1_detections[0].id[-1])
    current_detection = tless1_detections[0]
    for idx in range(1, len(tless1_detections)):
        if int(tless1_detections[idx].id[-1]) < current_idx:
            current_idx = tless1_detections[idx].id[-1]
            current_detection = tless1_detections[idx]

    return current_detection.results[0].pose


def get_increasing_weight(time, max_weight, percent, time_reach_percent):
    return max_weight * np.tanh(time * np.arctanh(percent) / time_reach_percent)


class ControllerBase:
    def __init__(self, params: AgimusControllerNodeParameters) -> None:
        self.params = params
        self.traj_buffer = TrajectoryBuffer()
        self.initialize_state_machine_attributes()
        self.rmodel, self.cmodel = get_pick_and_place_task_models()
        self.rdata = self.rmodel.createData()
        self.effector_frame_id = self.rmodel.getFrameId(
            self.params.ocp.effector_frame_name
        )
        self.nq = self.rmodel.nq
        self.nv = self.rmodel.nv
        self.nx = self.nq + self.nv
        self.ocp = OCPCrocoHPP(self.rmodel, self.cmodel, params.ocp)
        self.mpc_data = {}
        self.init_in_world_M_object = None
        self.in_world_M_object = None

        if self.params.use_ros_params:
            self.initialize_ros_attributes()

        if self.params.use_vision or self.params.use_vision_simulated:
            self.initialize_vision_attributes()
        self.target_translation_object_to_effector = None
        self.last_point = None
        self.pick_traj_last_pose = None
        self.start_visual_servoing_time = None
        self.do_visual_servoing = False
        self.in_world_M_prev_world = pin.XYZQUATToSE3(
            np.array([0.563, -0.166, 0.78, 0, 0, 1, 0])
        ).inverse()

    def initialize_state_machine_attributes(self):
        self.elapsed_time = None
        self.last_elapsed_time = None
        self.state_machine_timeline = [
            HPPStateMachine.WAITING_PICK_TRAJECTORY,
            HPPStateMachine.RECEIVING_PICK_TRAJECTORY,
            HPPStateMachine.WAITING_PLACE_TRAJECTORY,
            HPPStateMachine.RECEIVING_PLACE_TRAJECTORY,
            HPPStateMachine.WAITING_GOING_INIT_POSE_TRAJECTORY,
            HPPStateMachine.RECEIVING_GOING_INIT_POSE_TRAJECTORY,
        ]
        self.state_machine_timeline_idx = 0
        self.state_machine = self.state_machine_timeline[
            self.state_machine_timeline_idx
        ]
        self.point_attributes = [PointAttribute.Q, PointAttribute.V, PointAttribute.A]

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
        self.state_pub = rospy.Publisher("state", Int8, queue_size=1, tcp_nodelay=True)
        self.happypose_pose_pub = rospy.Publisher(
            "happypose_pose", Pose, queue_size=1, tcp_nodelay=True
        )
        self.first_robot_sensor_msg_received = False
        self.first_pose_ref_msg_received = True

    def initialize_vision_attributes(self):
        if self.params.use_vision:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
            # to avoid errors when looking at transforms too early in time
            time.sleep(0.5)
            self.vision_subscriber = rospy.Subscriber(
                "/labeled/detections", Detection2DArray, self.vision_callback
            )
        if self.params.use_vision_simulated:
            self.simulate_happypose_callback()
        if self.params.use_vision and self.params.use_vision_simulated:
            raise RuntimeError("use_vision and use_vision_simulated can't both be true")

    def simulate_happypose_callback(self):
        self.init_in_world_M_object = pin.XYZQUATToSE3(
            np.array([0.0, 0.0, 0.85, 0.0, 0.0, 0.0, 1.0])
        )

        self.in_world_M_object = pin.XYZQUATToSE3(
            np.array([-0.03, -0.03, 0.85, 0.0, 0.0, 0.0, 1.0])
        )

    def sensor_callback(self, sensor_msg):
        with self.mutex:
            self.sensor_msg = deepcopy(sensor_msg)
            if not self.first_robot_sensor_msg_received:
                self.first_robot_sensor_msg_received = True

    def vision_callback(self, vision_msg: Detection2DArray):
        if vision_msg.detections == []:
            return
        in_camera_pose_object = find_tracked_object(vision_msg.detections)
        image_timestamp = vision_msg.detections[0].header.stamp
        in_world_M_camera = self.tf_buffer.lookup_transform(
            target_frame="world",
            source_frame="camera_color_optical_frame",
            time=image_timestamp,
        )
        in_world_pose_object = tf2_geometry_msgs.do_transform_pose(
            in_camera_pose_object, in_world_M_camera
        )
        pose = in_world_pose_object.pose
        trans = pose.position
        rot = pose.orientation
        pose_array = [trans.x, trans.y, trans.z, rot.w, rot.x, rot.y, rot.z]
        in_prev_world_M_object = pin.XYZQUATToSE3(pose_array)
        self.in_world_M_object = self.in_world_M_prev_world * in_prev_world_M_object
        if self.init_in_world_M_object is None:
            self.init_in_world_M_object = self.in_world_M_object

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

    def wait_buffer_has_twice_horizon_points(self):
        while (
            self.traj_buffer.get_size(self.point_attributes)
            < 2 * self.params.ocp.horizon_size
        ):
            self.fill_buffer()

    def get_next_trajectory_point(self):
        raise RuntimeError("Not implemented")

    def update_state_machine(self):
        raise RuntimeError("Not implemented")

    def fill_buffer(self):
        point = self.get_next_trajectory_point()
        if point is not None:
            self.traj_buffer.add_trajectory_point(point)
        while point is not None:
            point = self.get_next_trajectory_point()
            if point is not None:
                self.traj_buffer.add_trajectory_point(point)

    def first_solve(self, x0):
        # retrieve horizon state and acc references
        horizon_points = self.traj_buffer.get_points(
            self.params.ocp.horizon_size, self.point_attributes
        )
        x_plan = np.zeros([self.params.ocp.horizon_size, self.nx])
        a_plan = np.zeros([self.params.ocp.horizon_size, self.nv])
        for idx_point, point in enumerate(horizon_points):
            x_plan[idx_point, :] = point.get_x_as_q_v()
            a_plan[idx_point, :] = point.a

        # First solve
        self.mpc = MPC(self.ocp, x_plan, a_plan, self.rmodel, self.cmodel)
        self.mpc.mpc_first_step(x_plan, a_plan, x0, self.params.ocp.horizon_size)
        self.next_node_idx = self.params.ocp.horizon_size
        if self.params.save_predictions_and_refs:
            self.mpc.create_mpc_data(self.params.ocp.use_constraints)
        _, u, k = self.mpc.get_mpc_output()
        return u, k

    def solve(self, x0):
        self.target_translation_object_to_effector = None
        self.fill_buffer()
        if self.traj_buffer.get_size(self.point_attributes) > 0:
            point = self.traj_buffer.get_points(1, self.point_attributes)[0]
            self.last_point = point
            new_x_ref = point.get_x_as_q_v()
        else:
            point = self.last_point
            new_x_ref = point.get_x_as_q_v()
        new_a_ref = point.a
        self.in_world_M_effector = get_ee_pose_from_configuration(
            self.rmodel,
            self.rdata,
            self.effector_frame_id,
            new_x_ref[: self.rmodel.nq],
        )
        # if last point of the pick trajectory is in horizon and we wanna use vision pose
        if (self.params.use_vision or self.params.use_vision_simulated) and (
            self.pick_traj_last_point_is_near(x0)
        ):
            self.update_effector_placement_with_vision()
            self.set_increasing_weight()
        else:
            self.mpc.ocp.set_last_running_model_placement_weight(0)
        self.mpc.mpc_step(x0, new_x_ref, new_a_ref, self.in_world_M_effector)
        if self.next_node_idx < self.mpc.whole_x_plan.shape[0] - 1:
            self.next_node_idx += 1

        if self.params.save_predictions_and_refs:
            self.mpc.fill_predictions_and_refs_arrays(self.params.ocp.use_constraints)
        _, u, k = self.mpc.get_mpc_output()
        return u, k

    def set_increasing_weight(self):
        visual_servoing_time = time.time() - self.start_visual_servoing_time
        gripper_weight_last_running_node = get_increasing_weight(
            max(visual_servoing_time - self.params.ocp.dt, 0.0),
            self.params.increasing_weights["max"] / self.params.ocp.dt,
            self.params.increasing_weights["percent"],
            self.params.increasing_weights["time_reach_percent"],
        )
        self.mpc.ocp.set_last_running_model_placement_cost(
            self.in_world_M_effector, gripper_weight_last_running_node
        )
        gripper_weight_terminal_node = get_increasing_weight(
            visual_servoing_time,
            self.params.increasing_weights["max"],
            self.params.increasing_weights["percent"],
            self.params.increasing_weights["time_reach_percent"],
        )
        self.mpc.ocp.set_ee_placement_weight(gripper_weight_terminal_node)

    def update_effector_placement_with_vision(self):
        in_object_rot_effector = (
            self.init_in_world_M_object.rotation.T * self.in_world_M_effector.rotation
        )
        self.target_translation_object_to_effector = (
            self.in_world_M_effector.translation
            - self.init_in_world_M_object.translation
        )

        self.in_world_M_effector.translation = (
            self.in_world_M_object.translation
            + self.target_translation_object_to_effector
        )
        self.in_world_M_effector.rotation = (
            self.in_world_M_object.rotation * in_object_rot_effector
        )

    def pick_traj_last_point_is_in_horizon(self):
        return (
            self.state_machine == HPPStateMachine.WAITING_PLACE_TRAJECTORY
            and self.traj_buffer.get_size(self.point_attributes) == 0
        )

    def pick_traj_last_point_is_near(self, x0):
        if (
            self.state_machine == HPPStateMachine.WAITING_PLACE_TRAJECTORY
            and self.pick_traj_last_pose is None
        ):
            last_pick_traj_point = self.traj_buffer.get_last_point(
                self.point_attributes
            )
            x = last_pick_traj_point.get_x_as_q_v()
            self.pick_traj_last_pose = get_ee_pose_from_configuration(
                self.rmodel,
                self.rdata,
                self.effector_frame_id,
                x[: self.rmodel.nq],
            ).translation
        current_pose = get_ee_pose_from_configuration(
            self.rmodel,
            self.rdata,
            self.effector_frame_id,
            x0[: self.rmodel.nq],
        ).translation

        if self.pick_traj_last_pose is None:
            return False
        else:
            if self.state_machine == HPPStateMachine.WAITING_PLACE_TRAJECTORY:
                if not self.do_visual_servoing:
                    self.do_visual_servoing = (
                        self.state_machine == HPPStateMachine.WAITING_PLACE_TRAJECTORY
                        and np.linalg.norm(self.pick_traj_last_pose - current_pose)
                        < self.params.start_visual_servoing_dist
                    )
                    if self.do_visual_servoing:
                        self.start_visual_servoing_time = time.time()
            else:
                self.do_visual_servoing = False
            return self.do_visual_servoing

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

        self.mpc_data["preds_xs"] = [xs]
        self.mpc_data["preds_us"] = [us]
        self.mpc_data["state_refs"] = [x_ref]
        self.mpc_data["translation_refs"] = [p_ref]
        self.mpc_data["control_refs"] = [u_ref]
        self.mpc_data["state"] = [self.state_machine.value]
        if self.params.use_constraints:
            collision_residuals = self.mpc.get_collision_residuals()
            self.mpc_data["coll_residuals"] = collision_residuals
        if self.in_world_M_object is not None:
            self.mpc_data["vision_refs"] = [
                np.array(self.in_world_M_object.translation)
            ]
            self.mpc_data["obj_trans_ee"] = []

    def fill_predictions_and_refs_arrays(self):
        xs, us = self.mpc.get_predictions()
        x_ref, p_ref, u_ref = self.mpc.get_reference()
        self.mpc_data["preds_xs"].append(xs)

        self.mpc_data["preds_us"].append(us)
        self.mpc_data["state_refs"].append(x_ref)
        self.mpc_data["translation_refs"].append(p_ref)
        self.mpc_data["control_refs"].append(u_ref)
        self.mpc_data["state"].append(self.state_machine.value)
        if self.init_in_world_M_object is not None:
            self.mpc_data["init_in_world_M_object"] = self.init_in_world_M_object
        if self.params.use_constraints:
            collision_residuals = self.mpc.get_collision_residuals()
            for coll_residual_key in collision_residuals.keys():
                self.mpc_data["coll_residuals"][coll_residual_key] += (
                    collision_residuals[coll_residual_key]
                )

        if "vision_refs" in self.mpc_data.keys():
            self.mpc_data["vision_refs"].append(
                np.array(self.in_world_M_object.translation)
            )

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

    def publish_vision_pose(self):
        if self.in_world_M_object is not None:
            in_world_pose_object = pin.SE3ToXYZQUAT(self.in_world_M_object)
            in_world_pose_object_ros = Pose()
            in_world_pose_object_ros.position.x = in_world_pose_object[0]
            in_world_pose_object_ros.position.y = in_world_pose_object[1]
            in_world_pose_object_ros.position.z = in_world_pose_object[2]
            in_world_pose_object_ros.orientation.x = in_world_pose_object[3]
            in_world_pose_object_ros.orientation.y = in_world_pose_object[4]
            in_world_pose_object_ros.orientation.z = in_world_pose_object[5]
            in_world_pose_object_ros.orientation.w = in_world_pose_object[6]
            self.happypose_pose_pub.publish(in_world_pose_object_ros)

    def run(self):
        self.wait_first_sensor_msg()
        self.wait_buffer_has_twice_horizon_points()
        sensor_msg = self.get_sensor_msg()
        start_compute_time = time.time()
        u, k = self.first_solve(self.get_x0_from_sensor_msg(sensor_msg))
        compute_time = time.time() - start_compute_time
        self.send(sensor_msg, u, k)
        self.publish_ocp_solve_time(compute_time)
        self.ocp_x0_pub.publish(sensor_msg)
        self.state_pub.publish(Int8(self.state_machine.value))
        self.publish_vision_pose()
        self.rate.sleep()
        atexit.register(self.exit_handler)
        self.run_timer = rospy.Timer(
            rospy.Duration(self.params.ocp.dt), self.run_callback
        )
        rospy.spin()

    def run_callback(self, *args):
        start_compute_time = time.time()
        self.update_state_machine()
        sensor_msg = self.get_sensor_msg()
        u, k = self.solve(self.get_x0_from_sensor_msg(sensor_msg))
        self.send(sensor_msg, u, k)
        compute_time = time.time() - start_compute_time
        self.publish_ocp_solve_time(compute_time)
        self.state_pub.publish(Int8(self.state_machine.value))
        self.ocp_x0_pub.publish(sensor_msg)
        self.publish_vision_pose()
