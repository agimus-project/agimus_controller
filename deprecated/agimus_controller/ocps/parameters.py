#!/usr/bin/env python3
import numpy as np


class OCPParameters:
    def __init__(self) -> None:
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
