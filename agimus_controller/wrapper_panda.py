# 2-Clause BSD License

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from os.path import dirname, join, abspath

import numpy as np
import pinocchio as pin
import hppfcl

import pybullet
from mim_robots.pybullet.wrapper import PinBulletWrapper
from pinocchio.robot_wrapper import RobotWrapper

from scenes import Scene

# This class is for unwrapping an URDF and converting it to a model. It is also possible to add objects in the model,
# such as a ball at a specific position.

RED = np.array([249, 136, 126, 125]) / 255


class PandaWrapper:
    def __init__(
        self,
        auto_col=False,
        capsule=False,
    ):
        """Create a wrapper for the robot panda.

        Args:
            auto_col (bool, optional): Include the auto collision in the collision model. Defaults to False.
            capsule (bool, optional): Transform the spheres and cylinder of the robot into capsules. Defaults to False.
        """

        # Importing the model
        pinocchio_model_dir = dirname(dirname(((str(abspath(__file__))))))
        model_path = join(pinocchio_model_dir, "robot_description")
        self._mesh_dir = join(model_path, "meshes")
        urdf_filename = "franka2.urdf"
        srdf_filename = "demo.srdf"
        self._urdf_model_path = join(join(model_path, "urdf"), urdf_filename)
        self._srdf_model_path = join(join(model_path, "srdf"), srdf_filename)

        # Color of the robot
        self._color = np.array([249, 136, 126, 255]) / 255

        # Boolean describing whether the auto-collisions are in the collision model or not
        self._auto_col = auto_col

        # Transforming the robot from cylinders/spheres to capsules
        self._capsule = capsule

    def __call__(self):
        """Create a robot.

        Returns:
            rmodel (pin.Model): Model of the robot
            cmodel (pin.GeometryModel): Collision model of the robot
            vmodel (pin.GeometryModel): Visual model of the robot
        """
        (
            self._rmodel,
            self._cmodel,
            self._vmodel,
        ) = pin.buildModelsFromUrdf(
            self._urdf_model_path, self._mesh_dir, pin.JointModelFreeFlyer()
        )

        q0 = pin.neutral(self._rmodel)

        # Locking the gripper
        jointsToLockIDs = [1, 9, 10]

        geom_models = [self._vmodel, self._cmodel]
        self._model_reduced, geometric_models_reduced = pin.buildReducedModel(
            self._rmodel,
            list_of_geom_models=geom_models,
            list_of_joints_to_lock=jointsToLockIDs,
            reference_configuration=q0,
        )

        self._vmodel_reduced, self._cmodel_reduced = (
            geometric_models_reduced[0],
            geometric_models_reduced[1],
        )

        # Modifying the collision model to transform the spheres/cylinders into capsules
        if self._capsule:
            self.transform_model_into_capsules()

        # Adding the auto-collisions in the collision model if required
        if self._auto_col:
            self._cmodel_reduced.addAllCollisionPairs()
            pin.removeCollisionPairs(
                self._model_reduced, self._cmodel_reduced, self._srdf_model_path
            )

        rdata = self._model_reduced.createData()
        cdata = self._cmodel_reduced.createData()
        q0 = pin.neutral(self._model_reduced)

        # Updating the models
        pin.framesForwardKinematics(self._model_reduced, rdata, q0)
        pin.updateGeometryPlacements(
            self._model_reduced, rdata, self._cmodel_reduced, cdata, q0
        )

        return (
            self._model_reduced,
            self._cmodel_reduced,
            self._vmodel_reduced,
        )

    def transform_model_into_capsules(self):
        """Modifying the collision model to transform the spheres/cylinders into capsules which makes it easier to have a fully constrained robot."""
        collision_model_reduced_copy = self._cmodel_reduced.copy()
        list_names_capsules = []

        # Going through all the goemetry objects in the collision model
        for geom_object in collision_model_reduced_copy.geometryObjects:
            if isinstance(geom_object.geometry, hppfcl.Cylinder):
                # Sometimes for one joint there are two cylinders, which need to be defined by two capsules for the same link.
                # Hence the name convention here.
                if (geom_object.name[:-4] + "capsule_0") in list_names_capsules:
                    name = geom_object.name[:-4] + "capsule_" + "1"
                else:
                    name = geom_object.name[:-4] + "capsule_" + "0"
                list_names_capsules.append(name)
                placement = geom_object.placement
                parentJoint = geom_object.parentJoint
                parentFrame = geom_object.parentFrame
                geometry = geom_object.geometry
                geom = pin.GeometryObject(
                    name,
                    parentFrame,
                    parentJoint,
                    hppfcl.Capsule(geometry.radius, geometry.halfLength),
                    placement,
                )
                geom.meshColor = RED
                self._cmodel_reduced.addGeometryObject(geom)
                self._cmodel_reduced.removeGeometryObject(geom_object.name)
            elif (
                isinstance(geom_object.geometry, hppfcl.Sphere)
                and "link" in geom_object.name
            ):
                self._cmodel_reduced.removeGeometryObject(geom_object.name)


class PandaRobot(PinBulletWrapper):
    """
    Pinocchio-PyBullet wrapper class for the KUKA LWR iiwa
    """

    def __init__(
        self,
        capsule=True,
        auto_col=False,
        qref=np.zeros(7),
        pos_robot=None,
        pos_obs=None,
        name_scene="box",
    ):
        # Load the robot

        # Create the robot wrapper in pinocchio.
        robot_wrapper = PandaWrapper(capsule=capsule, auto_col=auto_col)
        rmodel, cmodel, vmodel = robot_wrapper()

        self.scene = Scene(name_scene, obstacle_pose=pos_obs)
        rmodel, cmodel, self.TARGET_POSE1, self.TARGET_POSE2, self.q0 = self.scene.create_scene_from_urdf(
            rmodel,
            cmodel,
        )

        robot_full = RobotWrapper(rmodel, cmodel, vmodel)

        if pos_robot is None:
            pos_robot = [0.0, 0, 0.0]
            orn_robot = pybullet.getQuaternionFromEuler([0, 0, 0])

        pinocchio_model_dir = dirname(dirname(((str(abspath(__file__))))))
        model_path = join(pinocchio_model_dir, "robot_description")
        urdf_filename = "franka2.urdf"
        self._urdf_path = join(join(model_path, "urdf"), urdf_filename)

        if self.scene.urdf_filename is not None:
            self._urdf_path_obs = join(
                join(model_path, "urdf/obstacles"), self.scene.urdf_filename
            )

            pos_obs_quat = pin.SE3ToXYZQUATtuple(pos_obs)
            self.obstacleId = pybullet.loadURDF(
                self._urdf_path_obs,
                pos_obs_quat[:3],
                pos_obs_quat[3:],
                useFixedBase=True,
            )

        self.robotId = pybullet.loadURDF(
            self._urdf_path,
            pos_robot,
            orn_robot,
            useFixedBase=True,
        )
        pybullet.getBasePositionAndOrientation(self.robotId)

        # Query all the joints.
        num_joints = pybullet.getNumJoints(self.robotId)

        for ji in range(num_joints):
            pybullet.changeDynamics(
                self.robotId,
                ji,
                linearDamping=0.04,
                angularDamping=0.04,
                restitution=0.0,
                lateralFriction=0.5,
            )

        self.pin_robot = robot_full
        controlled_joints_names = [
            "panda2_joint1",
            "panda2_joint2",
            "panda2_joint3",
            "panda2_joint4",
            "panda2_joint5",
            "panda2_joint6",
            "panda2_joint7",
        ]

        self.base_link_name = "support_joint"
        self.end_eff_ids = []
        self.end_eff_ids.append(self.pin_robot.model.getFrameId("panda2_rightfinger"))
        self.nb_ee = len(self.end_eff_ids)
        self.joint_names = controlled_joints_names

        # Creates the wrapper by calling the super.__init__.
        super().__init__(
            self.robotId,
            self.pin_robot,
            controlled_joints_names,
            ["panda2_finger_joint1"],
            useFixedBase=True,
        )
        self.nb_dof = self.nv

    def forward_robot(self, q=None, dq=None):
        if q is None:
            q, dq = self.get_state()
        elif dq is None:
            raise ValueError("Need to provide q and dq or non of them.")

        self.pin_robot.forwardKinematics(q, dq)
        self.pin_robot.computeJointJacobians(q)
        self.pin_robot.framesForwardKinematics(q)
        self.pin_robot.centroidalMomentum(q, dq)

    def start_recording(self, file_name):
        self.file_name = file_name
        pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, self.file_name)

    def stop_recording(self):
        pybullet.stopStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, self.file_name)


if __name__ == "__main__":
    from wrapper_meshcat import MeshcatWrapper

    # Creating the robot
    robot_wrapper = PandaWrapper(capsule=True, auto_col=True)
    rmodel, cmodel, vmodel = robot_wrapper()
    rdata = rmodel.createData()
    cdata = cmodel.createData()
    # Generating the meshcat visualizer
    MeshcatVis = MeshcatWrapper()
    vis = MeshcatVis.visualize(
        robot_model=rmodel, robot_visual_model=cmodel, robot_collision_model=cmodel
    )
    # vis[0].display(pin.randomConfiguration(rmodel))
    vis[0].display(np.array([0.5] * 7))

    pin.computeCollisions(rmodel, rdata, cmodel, cdata, pin.neutral(rmodel), False)
    for k in range(len(cmodel.collisionPairs)):
        cr = cdata.collisionResults[k]
        cp = cmodel.collisionPairs[k]
        print(
            "collision pair:",
            cmodel.geometryObjects[cp.first].name,
            ",",
            cmodel.geometryObjects[cp.second].name,
            "- collision:",
            "Yes" if cr.isCollision() else "No",
        )
    q = pin.randomConfiguration(rmodel)
    vis[0].display(pin.randomConfiguration(rmodel))
