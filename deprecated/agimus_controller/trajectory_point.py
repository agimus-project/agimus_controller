from __future__ import annotations
import numpy as np
from enum import Enum


class PointAttribute(Enum):
    Q = 0
    V = 1
    A = 2
    TAU = 3
    COM_POS = 4
    COM_VEL = 5


class TrajectoryPoint:
    def __init__(self, time=0, nq=0, nv=0):
        self.q = np.zeros(nq) * np.nan
        self.v = np.zeros(nv) * np.nan
        self.a = np.zeros(nv) * np.nan
        self.tau = np.zeros(nv) * np.nan
        self.com_pos = np.zeros(3) * np.nan
        self.com_vel = np.zeros(3) * np.nan
        self.op_pos = {}
        self.op_vel = {}
        self.nq = nq
        self.nv = nv
        self.time = time
        self.attribute_validation_dict = self.get_attribute_validation_dict()

    def get_attribute_validation_dict(self):
        attribute_validation_dict = {}
        attribute_validation_dict[PointAttribute.Q] = self.q_is_valid
        attribute_validation_dict[PointAttribute.V] = self.v_is_valid
        attribute_validation_dict[PointAttribute.A] = self.a_is_valid
        attribute_validation_dict[PointAttribute.TAU] = self.tau_is_valid
        attribute_validation_dict[PointAttribute.COM_POS] = self.com_pos_is_valid
        attribute_validation_dict[PointAttribute.COM_VEL] = self.com_vel_is_valid
        return attribute_validation_dict

    def resize(self, nq, nv):
        self.q = np.zeros(nq) * np.nan
        self.v = np.zeros(nv) * np.nan
        self.a = np.zeros(nv) * np.nan
        self.tau = np.zeros(nv) * np.nan
        self.nq = nq
        self.nv = nv

    def get_x_as_q_v(self):
        return np.concatenate([self.q, self.v])

    def attribute_is_valid(self, attribute: PointAttribute):
        return self.attribute_validation_dict[attribute]()

    def q_is_valid(self):
        return not np.array_equal(self.q, np.zeros(self.nq) * np.nan)

    def v_is_valid(self):
        return not np.array_equal(self.v, np.zeros(self.nv) * np.nan)

    def a_is_valid(self):
        return not np.array_equal(self.a, np.zeros(self.nv) * np.nan)

    def tau_is_valid(self):
        return not np.array_equal(self.tau, np.zeros(self.nv) * np.nan)

    def com_pos_is_valid(self):
        return not np.array_equal(self.com_pos, np.zeros(3) * np.nan)

    def com_vel_is_valid(self):
        return not np.array_equal(self.com_vel, np.zeros(3) * np.nan)
