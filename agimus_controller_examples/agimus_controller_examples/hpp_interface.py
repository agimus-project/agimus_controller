#!/usr/bin/env python
#
#  Copyright 2020 CNRS
#
#  Author: Florent Lamiraux
#
# Start hppcorbaserver before running this script
#

import numpy as np
from hpp.corbaserver import loadServerPlugin
from agimus_controller.trajectory import TrajectoryPoint
from agimus_controller_examples.hpp_panda.planner import Planner as PandaPlanner
from agimus_controller_examples.hpp_panda.scenes import Scene
from agimus_controller_examples.utils.set_models_and_mpc import get_panda_models


class Sphere(object):
    rootJointType = "freeflyer"
    packageName = "hpp_environments"
    urdfName = "construction_set/sphere"
    urdfSuffix = ""
    srdfSuffix = ""


class Ground(object):
    rootJointType = "anchor"
    packageName = "hpp_environments"
    urdfName = "construction_set/ground"
    urdfSuffix = ""
    srdfSuffix = ""


class HppInterface:
    def __init__(self):
        self.trajectory = []
        self.viewer = None

    def set_panda_planning(self, q_init, q_goal, use_gepetto_gui=False):
        self.q_init = q_init
        self.q_goal = q_goal
        loadServerPlugin("corbaserver", "manipulation-corba.so")
        self.T = 20
        self.robot_models = get_panda_models("agimus_demo_03_mpc_dummy_traj")
        """
        panda_params = PandaRobotModelParameters()
        panda_params.collision_as_capsule = True
        panda_params.self_collision = True
        self.robot_wrapper = PandaRobotModel.load_model(
            params=panda_params,
            env=Path(__file__).resolve().parent / "resources" / "panda_env.yaml",
        )"""

        self.scene = Scene("wall", self.q_init)
        (
            self.robot_models.robot_model,
            self.robot_models.collision_model,
            self.target,
            self.target2,
            _,
        ) = self.scene.create_scene_from_urdf(
            self.robot_models.robot_model, self.robot_models.collision_model
        )
        self.planner = PandaPlanner(self.robot_models, self.scene, self.T)
        self.planner.setup_planner(q_init, q_goal, use_gepetto_gui)
        _, _, self.X = self.planner.solve_and_optimize()
        self.planner._ps.optimizePath(self.planner._ps.numberPaths() - 1)
        self.ps = self.planner._ps
        if use_gepetto_gui:
            self.viewer = self.planner._v
        self.problem = self.ps.client.problem

    def get_problem_solver(self):
        return self.ps

    def get_viewer(self):
        return self.viewer

    def get_hpp_x_a_planning(self, DT):
        nq = self.robot_models.robot_model.nq
        hpp_path = self.problem.getPath(self.ps.numberPaths() - 1)
        path = hpp_path.pathAtRank(0)
        T = int(np.round(path.length() / DT))
        x_plan, a_plan, subpath = self.get_xplan_aplan(T, path, nq)
        self.trajectory = subpath
        whole_traj_T = T
        for path_idx in range(1, hpp_path.numberPaths()):
            path = hpp_path.pathAtRank(path_idx)
            T = int(np.round(path.length() / DT))
            if T == 0:
                continue
            subpath_x_plan, subpath_a_plan, subpath = self.get_xplan_aplan(T, path, nq)
            x_plan = np.concatenate([x_plan, subpath_x_plan], axis=0)
            a_plan = np.concatenate([a_plan, subpath_a_plan], axis=0)
            self.trajectory += subpath
            whole_traj_T += T
        return x_plan, a_plan, whole_traj_T

    def get_xplan_aplan(self, T, path, nq):
        """Return x_plan the state and a_plan the acceleration of hpp's trajectory."""
        x_plan = np.zeros([T, 2 * nq])
        a_plan = np.zeros([T, nq])
        subpath = []
        trajectory_point = TrajectoryPoint()
        trajectory_point.q = np.zeros(nq)
        trajectory_point.v = np.zeros(nq)
        trajectory_point.a = np.zeros(nq)
        subpath = [trajectory_point]
        if T == 0:
            pass
        elif T == 1:
            time = path.length()
            q_t = np.array(path.call(time)[0][:nq])
            v_t = np.array(path.derivative(time, 1)[:nq])
            x_plan[0, :] = np.concatenate([q_t, v_t])
            a_t = np.array(path.derivative(time, 2)[:nq])
            a_plan[0, :] = a_t
            subpath[0].q[:] = q_t[:]
            subpath[0].v[:] = v_t[:]
            subpath[0].a[:] = a_t[:]
        else:
            total_time = path.length()
            subpath = [TrajectoryPoint(t, nq, nq) for t in range(T)]
            for iter in range(T):
                iter_time = total_time * iter / (T - 1)  # iter * DT
                q_t = np.array(path.call(iter_time)[0][:nq])
                v_t = np.array(path.derivative(iter_time, 1)[:nq])
                x_plan[iter, :] = np.concatenate([q_t, v_t])
                a_t = np.array(path.derivative(iter_time, 2)[:nq])
                a_plan[iter, :] = a_t
                subpath[iter].q[:] = q_t[:]
                subpath[iter].v[:] = v_t[:]
                subpath[iter].a[:] = a_t[:]
        return x_plan, a_plan, subpath

    def get_trajectory_point(self, index):
        return self.trajectory[index]

    def get_panda_q_init_q_goal(self):
        q_init = [
            0.0011252369260450479,
            0.0006049034973703016,
            -0.0008132459264533765,
            -1.573963485445721,
            -0.0034107252065869176,
            1.5572160817164562,
            0.7840735791072078,
        ]

        q_goal = [
            -0.1696519066916446,
            0.4713501253198094,
            -0.11462830134177399,
            -2.081133924808195,
            0.0929635353746508,
            2.547820269262958,
            0.43640470497878087,
        ]
        return q_init, q_goal
