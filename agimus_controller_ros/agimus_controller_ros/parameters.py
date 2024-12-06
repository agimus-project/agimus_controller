#!/usr/bin/env python3
import numpy as np
from rclpy.node import Node


class OCPParametersROS(Node):
    def __init__(self) -> None:
        super().__init__("params_node")
        self.dt = None
        self.horizon_size = None
        self.armature = None
        self.gripper_weight = None
        self.state_weight = None
        self.control_weight = None
        self.max_iter = None
        self.max_qp_iter = None
        self.use_constraints = None
        self.effector_frame_name = None
        self.activate_callback = None
        self.increasing_weights = None

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
        self.increasing_weights = params_dict["increasing_weights"]

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
        params["increasing_weights"] = self.increasing_weights
        return params

    def set_parameters_from_ros(self):
        self.declare_parameter("ocp/dt", 0.01)
        self.dt = self.get_parameter("ocp/dt").get_parameter_value().double_value
        self.declare_parameter("ocp/horizon_size", 20)
        self.horizon_size = (
            self.get_parameter("ocp/horizon_size").get_parameter_value().double_value
        )
        self.declare_parameter("ocp/armature", [])
        self.armature = (
            self.get_parameter("ocp/armature").get_parameter_value().double_array_value
        )

        self.declare_parameter("ocp/gripper_weight", 0.0)
        self.gripper_weight = (
            self.get_parameter("ocp/gripper_weight").get_parameter_value().double_value
        )

        self.declare_parameter("ocp/state_weight", 0.0)
        self.state_weight = (
            self.get_parameter("ocp/state_weight").get_parameter_value().double_value
        )

        self.declare_parameter("ocp/control_weight", 0.0)
        self.control_weight = (
            self.get_parameter("ocp/control_weight").get_parameter_value().double_value
        )

        self.declare_parameter("ocp/max_iter", 0)
        self.max_iter = (
            self.get_parameter("ocp/max_iter").get_parameter_value().integer_value
        )

        self.declare_parameter("ocp/max_qp_iter", 0)
        self.max_qp_iter = (
            self.get_parameter("ocp/max_qp_iter").get_parameter_value().integer_value
        )

        self.declare_parameter("ocp/use_constraints", False)
        self.use_constraints = (
            self.get_parameter("ocp/use_constraints").get_parameter_value().bool_value
        )

        self.declare_parameter("ocp/effector_frame_name", "")
        self.effector_frame_name = (
            self.get_parameter("ocp/effector_frame_name")
            .get_parameter_value()
            .string_value
        )
        self.declare_parameter("ocp/activate_callback", False)
        self.activate_callback = (
            self.get_parameter("ocp/activate_callback").get_parameter_value().bool_value
        )
        self.increasing_weights = {}
        self.declare_parameter("ocp/increasing_weights/max", 0.0)
        self.increasing_weights["max"] = (
            self.get_parameter("ocp/increasing_weights/max")
            .get_parameter_value()
            .double_value
        )
        self.declare_parameter("ocp/increasing_weights/percent", 0.0)

        self.increasing_weights["percent"] = (
            self.get_parameter("ocp/increasing_weights/percent")
            .get_parameter_value()
            .double_value
        )
        self.declare_parameter("ocp/increasing_weights/time_reach_percent", 0.0)
        self.increasing_weights["time_reach_percent"] = (
            self.get_parameter("ocp/increasing_weights/time_reach_percent")
            .get_parameter_value()
            .double_value
        )


class AgimusControllerNodeParameters(Node):
    def __init__(self) -> None:
        super().__init__("mpc_params")
        self.save_predictions_and_refs = None
        self.rate = None
        self.use_vision = None
        self.use_vision_simulated = None
        self.start_visual_servoing_dist = None
        self.ocp = OCPParametersROS()
        self.use_ros_params = None

    def set_parameters_from_ros(self):
        self.declare_parameter("save_predictions_and_refs", False)
        self.save_predictions_and_refs = (
            self.get_parameter("save_predictions_and_refs")
            .get_parameter_value()
            .string_value
        )
        self.declare_parameter("rate", 0)
        self.rate = self.get_parameter("rate").get_parameter_value().integer_value
        self.ocp.set_parameters_from_ros()
        self.use_ros_params = True

    def set_parameters_from_dict(self, params_dict):
        self.save_predictions_and_refs = params_dict["save_predictions_and_refs"]
        self.rate = params_dict["rate"]
        self.use_vision = params_dict["use_vision"]

        self.ocp.set_parameters_from_dict(params_dict["ocp"])
        self.use_ros_params = False

    def get_dict(self):
        params = {}
        params["save_predictions_and_refs"] = self.save_predictions_and_refs
        params["rate"] = self.rate
        params["ocp"] = self.ocp.get_dict()
        return params
