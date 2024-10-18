#!/usr/bin/env python3
import rospy
import numpy as np


class OCPParameters:
    def __init__(self, use_ros_params=True, params_dict=None) -> None:
        if use_ros_params:
            self.set_parameters_from_ros()
        elif params_dict is not None:
            self.set_parameters_from_dict(params_dict)
        else:
            raise RuntimeError("no parameters given for the controller")

    def set_parameters_from_ros(self):
        self.dt = rospy.get_param("ocp/dt", 0.01)
        self.horizon_size = rospy.get_param("ocp/horizon_size", 100)
        self.armature = np.array(
            rospy.get_param("ocp/armature", [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        )
        self.gripper_weight = rospy.get_param("ocp/gripper_weight", 10000)
        self.state_weight = rospy.get_param("ocp/state_weight", 10)
        self.control_weight = rospy.get_param("ocp/control_weight", 0.001)
        self.max_iter = rospy.get_param("ocp/max_iter", 1)
        self.max_qp_iter = rospy.get_param("ocp/max_qp_iter", 100)
        self.use_constraints = rospy.get_param("ocp/use_constraints", False)
        self.effector_frame_name = rospy.get_param(
            "ocp/effector_frame_name", "panda_hand_tcp"
        )
        self.activate_callback = rospy.get_param("ocp/activate_callback", False)

    def set_parameters_from_dict(self, params_dict):
        self.dt = params_dict["dt"]
        self.horizon_size = params_dict["horizon_size"]
        self.armature = np.array(params_dict["armature"])
        self.gripper_weight = params_dict["gripper_weight"]
        self.state_weight = params_dict["state_weight"]
        self.control_weight = params_dict["control_weight"]
        self.max_iter = params_dict["max_iter"]
        self.max_qp_iter = params_dict["max_qp_iter"]
        self.use_constraints = params_dict["use_constraints"]
        self.effector_frame_name = params_dict["effector_frame_name"]
        self.activate_callback = params_dict["activate_callback"]

    def get_dict(self):
        params = {}
        params["dt"] = self.dt
        params["horizon_size"] = self.horizon_size
        params["armature"] = self.armature
        params["gripper_weight"] = self.gripper_weight
        params["state_weight"] = self.state_weight
        params["control_weight"] = self.control_weight
        params["max_iter"] = self.max_iter
        params["max_qp_iter"] = self.max_qp_iter
        params["use_constraints"] = self.use_constraints
        params["effector_frame_name"] = self.effector_frame_name
        params["activate_callback"] = self.activate_callback
        return params


class AgimusControllerNodeParameters:
    def __init__(self, use_ros_params=True, params_dict=None) -> None:
        if use_ros_params:
            self.set_parameters_from_ros()
        elif params_dict is not None:
            self.set_parameters_from_dict(params_dict)
        else:
            raise RuntimeError("no parameters given for the controller")
        self.ocp = OCPParameters(use_ros_params, params_dict)
        self.use_ros_params = use_ros_params

    def set_parameters_from_ros(self):
        self.save_predictions_and_refs = rospy.get_param(
            "save_predictions_and_refs", False
        )
        self.rate = rospy.get_param("rate", 100)
        self.use_vision = rospy.get_param("use_vision", False)
        self.use_vision_simulated = rospy.get_param("use_vision_simulated", False)
        self.start_visual_servoing_dist = rospy.get_param(
            "start_visual_servoing_dist", 0.03
        )
        self.increasing_weights = rospy.get_param("increasing_weights", [])

    def set_parameters_from_dict(self, params_dict):
        self.save_predictions_and_refs = params_dict["save_predictions_and_refs"]
        self.rate = params_dict["rate"]
        self.use_vision = params_dict["use_vision"]
        self.use_vision_simulated = params_dict["use_vision_simulated"]
        self.start_visual_servoing_dist = params_dict["start_visual_servoing_dist"]
        self.increasing_weights = params_dict["increasing_weights"]
        self.ocp = OCPParameters(False, params_dict["ocp"])

    def get_dict(self):
        params = {}
        params["save_predictions_and_refs"] = self.save_predictions_and_refs
        params["rate"] = self.rate
        params["use_vision"] = self.use_vision
        params["use_vision_simulated"] = self.use_vision_simulated
        params["start_visual_servoing_dist"] = self.start_visual_servoing_dist
        params["increasing_weights"] = self.increasing_weights
        params["ocp"] = self.ocp.get_dict()
        return params
