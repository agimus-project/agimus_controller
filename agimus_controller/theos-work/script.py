#!/usr/bin/env python
#
#  Copyright 2020 CNRS
#
#  Author: Florent Lamiraux
#
# Start hppcorbaserver before running this script
#

import os
from argparse import ArgumentParser
from math import pi, fabs
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

parser = ArgumentParser()
parser.add_argument("-N", default=20, type=int)
args = parser.parse_args()
loadServerPlugin("corbaserver", "manipulation-corba.so")
Client().problem.resetProblem()
Robot.urdfFilename = (
    "package://example-robot-data/robots/ur_description/urdf/ur3_gripper.urdf"
)
Robot.srdfFilename = (
    "package://example-robot-data/robots/ur_description/srdf/ur3_gripper.srdf"
)
dir = os.getenv("PWD")


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
p = ps.client.basic.problem.getProblem()
r = p.robot()
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
## Contact surfaces
shapesPerObject = [[] for o in objects]
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
q_init = [pi / 6, -pi / 2, pi / 2, 0, 0, 0, -0.2, 0, 0.02, 0, 0, 0, 1]
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

# Uncomment to help M-RRT pathplanner
# for e in ['ur3/gripper > sphere0/handle | f_ls',
#           'ur3/gripper > sphere1/handle | f_ls'] :
#  cg.setWeight(e, 100)
# for e in ['ur3/gripper < sphere0/handle | 0-0_ls',
#          'ur3/gripper < sphere1/handle | 0-1_ls'] :
#  cg.setWeight(e, 100)
ps.selectPathValidation("Dichotomy", 0)
# ps.addPathOptimizer("SplineGradientBased_bezier1")
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

ps.setInitialConfig(q_init)
ps.addGoalConfig(q_goal)
ps.setMaxIterPathPlanning(5000)
# Run benchmark
#
import datetime as dt

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
    print(f"Success rate: {success/ args.N * 100}%")
    if success > 0:
        print(f"Average time per success: {totalTime.total_seconds()/success}")
        print(f"Average number nodes per success: {totalNumberNodes/success}")

##### start croco script
if __name__ == "__main__":
    from hpp.corbaserver import wrap_delete
    from .croco_hpp import *

    ball_init_pose = [-0.2, 0, 0.02, 0, 0, 0, 1]
    chc = CrocoHppConnection(ps, "ur5", vf, ball_init_pose)
    start = time.time()
    chc.prob.set_costs(10**4.5, 100, 10**-3.5, 0, 0)
    chc.search_best_costs(chc.prob.nb_paths - 1, False, False, True)
    # chc.do_mpc(chc.prob.nb_paths - 1, 100)
    end = time.time()
    print("search duration ", end - start)
    with open("datas.npy", "wb") as f:
        np.save(f, chc.prob.hpp_paths[0].x_plan)
        np.save(f, chc.prob.hpp_paths[1].x_plan)

"""
from hpp.gepetto import PathPlayer
v =vf.createViewer()
pp = PathPlayer (v)"""
