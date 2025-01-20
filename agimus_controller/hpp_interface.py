#!/usr/bin/env python
#
#  Copyright 2020 CNRS
#
#  Author: Florent Lamiraux
#
# Start hppcorbaserver before running this script
#

from pathlib import Path
import datetime as dt
import numpy as np
from argparse import ArgumentParser
from math import pi
from hpp.corbaserver.manipulation import (
    Client,
    ConstraintGraph,
    Rule,
    ConstraintGraphFactory,
    ProblemSolver,
    Constraints,
)
from hpp.corbaserver.manipulation.ur5 import Robot
from hpp.gepetto.manipulation import ViewerFactory
from hpp.corbaserver import loadServerPlugin
from hpp_idl.hpp import Equality, EqualToZero
from agimus_controller.trajectory_point import TrajectoryPoint
from agimus_controller.hpp_panda.planner import Planner as PandaPlanner
from agimus_controller.hpp_panda.scenes import Scene
from agimus_controller.robot_model.panda_model import (
    PandaRobotModel,
    PandaRobotModelParameters,
)
from agimus_controller.robot_model.ur3_model import UR3RobotModel


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

    def set_ur3_problem_solver(self, ur3_robot: UR3RobotModel):
        parser = ArgumentParser()
        parser.add_argument("-N", default=20, type=int)
        args = parser.parse_args()
        self.robot_wrapper = ur3_robot
        loadServerPlugin("corbaserver", "manipulation-corba.so")

        Client().problem.resetProblem()
        Robot.urdfFilename = (
            "package://example-robot-data/robots/ur_description/urdf/ur3_gripper.urdf"
        )
        Robot.srdfFilename = (
            "package://example-robot-data/robots/ur_description/srdf/ur3_gripper.srdf"
        )

        nSphere = 1
        robot = Robot("ur3-spheres", "ur3")  # ,rootJointType="anchor"
        ps = ProblemSolver(robot)
        ps.setErrorThreshold(1e-4)
        ps.setMaxIterProjection(40)
        vf = ViewerFactory(ps)
        ps.loadPlugin("spline-gradient-based.so")
        # Change bounds of robots to increase workspace and avoid some collisions
        robot.setJointBounds("ur3/shoulder_pan_joint", [-pi, 4])
        robot.setJointBounds("ur3/shoulder_lift_joint", [-pi, 0])
        robot.setJointBounds("ur3/elbow_joint", [-2.6, 2.6])
        vf.loadEnvironmentModel(Ground, "ground")
        objects = list()
        r = ps.client.basic.problem.getProblem().robot()
        for i in range(nSphere):
            vf.loadObjectModel(Sphere, "sphere{0}".format(i))
            robot.setJointBounds(
                "sphere{0}/root_joint".format(i),
                [
                    -1.0,
                    1.0,
                    -1.0,
                    1.0,
                    -0.1,
                    1.0,
                    -1.0001,
                    1.0001,
                    -1.0001,
                    1.0001,
                    -1.0001,
                    1.0001,
                    -1.0001,
                    1.0001,
                ],
            )
            objects.append("sphere{0}".format(i))
        ## Gripper
        #
        grippers = ["ur3/gripper"]
        ## Handles
        #
        handlesPerObject = [["sphere{0}/handle".format(i)] for i in range(nSphere)]
        contactsPerObject = [[] for i in range(nSphere)]
        ## Constraints
        #
        for i in range(nSphere):
            # Change mask of sphere handle
            o = objects[i]
            h = r.getHandle(o + "/handle")
            h.setMask([True, True, True, False, True, True])
            # placement constraint
            placementName = "place_sphere{0}".format(i)
            ps.createTransformationConstraint(
                placementName,
                "",
                "sphere{0}/root_joint".format(i),
                [0, 0, 0.02, 0, 0, 0, 1],
                [False, False, True, True, True, False],
            )
            ps.setConstantRightHandSide(placementName, True)
            # placement complement constraint
            ps.createTransformationConstraint(
                placementName + "/complement",
                "",
                "sphere{0}/root_joint".format(i),
                [0, 0, 0.02, 0, 0, 0, 1],
                [True, True, False, False, False, True],
            )
            ps.setConstantRightHandSide(placementName + "/complement", False)
            # combination of placement and complement
            ps.createLockedJoint(
                placementName + "/hold",
                "sphere{0}/root_joint".format(i),
                [0, 0, 0.02, 0, 0, 0, 1],
                [Equality, Equality, EqualToZero, EqualToZero, EqualToZero, Equality],
            )
            ps.registerConstraints(
                placementName, placementName + "/complement", placementName + "/hold"
            )
            preplacementName = "preplace_sphere{0}".format(i)
            ps.createTransformationConstraint(
                preplacementName,
                "",
                "sphere{0}/root_joint".format(i),
                [0, 0, 0.1, 0, 0, 0, 1],
                [False, False, True, True, True, False],
            )
            ps.setConstantRightHandSide(preplacementName, True)
        q_goal = [pi / 6, -pi / 2, pi / 2, 0, 0, 0, -0.3, 0, 0.02, 0, 0, 0, 1]
        lang = "py"
        if lang == "cxx":
            rules = [
                Rule(grippers, [""], True),
                Rule(grippers, ["sphere0/handle"], True),
                Rule(grippers, ["sphere1/handle"], True),
            ]
            cg = ConstraintGraph.buildGenericGraph(
                robot=robot,
                name="manipulation",
                grippers=grippers,
                objects=objects,
                handlesPerObjects=handlesPerObject,
                shapesPerObjects=contactsPerObject,
                envNames=[],
                rules=rules,
            )
        if lang == "py":
            cg = ConstraintGraph(robot, "manipulation")
            factory = ConstraintGraphFactory(cg)
            factory.setGrippers(grippers)
            factory.setObjects(objects, handlesPerObject, contactsPerObject)
            factory.generate()

        ps.setRandomSeed(1)
        ps.selectPathValidation("Dichotomy", 0)
        ps.addPathOptimizer("SimpleTimeParameterization")
        ps.setParameter("SimpleTimeParameterization/maxAcceleration", 1.0)
        ps.setParameter("SimpleTimeParameterization/order", 2)
        ps.setParameter("SimpleTimeParameterization/safety", 0.9)

        for i in range(nSphere):
            e = "ur3/gripper > sphere{}/handle | f_23".format(i)
            cg.addConstraints(
                edge=e,
                constraints=Constraints(
                    numConstraints=[
                        "place_sphere{}/complement".format(i),
                    ]
                ),
            )
            e = "ur3/gripper < sphere{}/handle | 0-{}_32".format(i, i)
            cg.addConstraints(
                edge=e,
                constraints=Constraints(
                    numConstraints=[
                        "place_sphere{}/complement".format(i),
                    ]
                ),
            )

        # need to set path projector due to implicit constraints added above
        ps.selectPathProjector("Progressive", 0.01)

        cg.initialize()
        q_init = self.robot_wrapper.get_default_full_configuration().tolist()
        ps.setInitialConfig(q_init)
        ps.addGoalConfig(q_goal)
        ps.setMaxIterPathPlanning(5000)
        # Run benchmark
        #
        totalTime = dt.timedelta(0)
        totalNumberNodes = 0
        success = 0
        # ps.setParameter("StatesPathFinder/nTriesUntilBacktrack", 10)
        for i in range(args.N):
            ps.clearRoadmap()
            ps.resetGoalConfigs()
            ps.setInitialConfig(q_init)
            ps.addGoalConfig(q_goal)
            try:
                t1 = dt.datetime.now()
                ps.solve()
                t2 = dt.datetime.now()
            except Exception as e:
                print(f"Failed to plan path: {e}")
            else:
                success += 1
                totalTime += t2 - t1
                print(t2 - t1)
                n = ps.numberNodes()
                totalNumberNodes += n
                print("Number nodes: " + str(n))
        if args.N != 0:
            print("#" * 20)
            print(f"Number of rounds: {args.N}")
            print(f"Number of successes: {success}")
            print(f"Success rate: {success / args.N * 100}%")
            if success > 0:
                print(
                    f"Average time per success: {totalTime.total_seconds() / success}"
                )
                print(f"Average number nodes per success: {totalNumberNodes / success}")
        self.ps = ps
        self.problem = ps.client.basic.problem
        self.viewer = vf.createViewer()

    def set_panda_planning(self, q_init, q_goal, use_gepetto_gui=False):
        self.q_init = q_init
        self.q_goal = q_goal
        loadServerPlugin("corbaserver", "manipulation-corba.so")
        self.T = 20
        panda_params = PandaRobotModelParameters()
        panda_params.collision_as_capsule = True
        panda_params.self_collision = True
        self.robot_wrapper = PandaRobotModel.load_model(
            params=panda_params,
            env=Path(__file__).resolve().parent / "resources" / "panda_env.yaml",
        )
        self.scene = Scene("wall", self.q_init)
        (
            self.robot_wrapper._rmodel,
            self.robot_wrapper._rcmodel,
            self.target,
            self.target2,
            _,
        ) = self.scene.create_scene_from_urdf(
            self.robot_wrapper._rmodel, self.robot_wrapper._rcmodel
        )
        self.planner = PandaPlanner(self.robot_wrapper, self.scene, self.T)
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
        nq = self.robot_wrapper._rmodel.nq
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
