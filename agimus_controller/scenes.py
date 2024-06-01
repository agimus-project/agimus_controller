from os.path import dirname, join, abspath


import numpy as np
import pinocchio as pin

import hppfcl

YELLOW_FULL = np.array([1, 1, 0, 1.0])


class Scene:
    def __init__(
        self,
        name_scene: str,
        obstacle_pose: pin.SE3,
    ) -> None:

        self._name_scene = name_scene
        self.obstacle_pose = obstacle_pose
        if self._name_scene == "box":
            self.urdf_filename = "big_box.urdf"
            self._TARGET_POSE1 = pin.SE3(pin.utils.rotate("x", np.pi), np.array([0, -0.4, 0.85]))
            self._TARGET_POSE2 = pin.SE3(pin.utils.rotate("x", np.pi), np.array([0, 0.15, 0.85]))
            self._q0 = np.array(
                [6.2e-01, 1.7e00, 1.5e00, 6.9e-01, -1.3e00, 1.1e00, 1.5e-01]
            )
        else:
            raise NotImplementedError(f"The input {self._name_scene} is not implemented.")
        
    def create_scene_from_urdf(
        self,
        rmodel: pin.Model,
        cmodel: pin.Model,
    ):
        """Create a scene amond the ones : "box"

        Args:
            rmodel (pin.Model): robot model
            cmodel (pin.Model): collision model of the robot
            name_scene (str): name of the scene
            obstacle_pose (pin.SE3): pose in the world frame of the obstacle as a whole.

        """

        if self._name_scene == "box":
            model, collision_model, visual_model = self.load_obstacle_urdf(
                self.urdf_filename
            )
            self._rmodel, self._cmodel = pin.appendModel(
                rmodel,
                model,
                cmodel,
                collision_model,
                0,
                self.obstacle_pose,
            )

        self._add_collision_pairs_urdf()
        return self._cmodel, self._TARGET_POSE1, self._TARGET_POSE2, self._q0

    def create_scene(self, rmodel: pin.Model, cmodel: pin.Model, name_scene: str):
        """Create a scene amond the ones : "box", "wall", "ball".

        Args:
            rmodel (pin.Model): robot model
            cmodel (pin.Model): collision model of the robot
            name_scene (str): name of the scene
        """

        self._name_scene = name_scene
        self._cmodel = cmodel
        self._rmodel = rmodel

        self._target = pin.SE3.Identity()
        if self._name_scene == "box":
            self._target = pin.SE3(
                pin.utils.rotate("x", np.pi), np.array([0.0, 0.2, 0.8])
            )
            self._q0 = np.array(
                [6.2e-01, 1.7e00, 1.5e00, 6.9e-01, -1.3e00, 1.1e00, 1.5e-01]
            )
            OBSTACLE_HEIGHT = 0.85
            OBSTACLE_X = 2.0e-1
            OBSTACLE_Y = 0.5e-2
            OBSTACLE_Z = 0.5
            obstacles = [
                (
                    "obstacle1",
                    hppfcl.Box(OBSTACLE_X, OBSTACLE_Y, OBSTACLE_Z),
                    pin.SE3(
                        pin.utils.rotate("y", np.pi / 2),
                        np.array([-0.0, -0.1, OBSTACLE_HEIGHT]),
                    ),
                ),
                (
                    "obstacle2",
                    hppfcl.Box(OBSTACLE_X, OBSTACLE_Y, OBSTACLE_Z),
                    pin.SE3(
                        pin.utils.rotate("y", np.pi / 2),
                        np.array([-0.0, 0.4, OBSTACLE_HEIGHT]),
                    ),
                ),
                (
                    "obstacle3",
                    hppfcl.Box(OBSTACLE_X, OBSTACLE_Y, OBSTACLE_Z),
                    pin.SE3(
                        pin.utils.rotate("y", np.pi / 2)
                        @ pin.utils.rotate("x", np.pi / 2),
                        np.array([0.25, 0.15, OBSTACLE_HEIGHT]),
                    ),
                ),
                (
                    "obstacle4",
                    hppfcl.Box(OBSTACLE_X, OBSTACLE_Y, OBSTACLE_Z),
                    pin.SE3(
                        pin.utils.rotate("y", np.pi / 2)
                        @ pin.utils.rotate("x", np.pi / 2),
                        np.array([-0.25, 0.15, OBSTACLE_HEIGHT]),
                    ),
                ),
            ]
        elif self._name_scene == "ball":
            self._q0 = pin.neutral(self._rmodel)
            self._target.translation = np.array([0, -0.4, 1.5])
            OBSTACLE_RADIUS = 1.5e-1
            OBSTACLE_POSE = pin.SE3.Identity()
            OBSTACLE_POSE.translation = np.array([0.25, -0.4, 1.5])
            obstacles = [("obstacle1", hppfcl.Sphere(OBSTACLE_RADIUS), OBSTACLE_POSE)]
        elif self._name_scene == "wall":
            self._target = pin.SE3(
                pin.utils.rotate("x", np.pi), np.array([0.0, 0.2, 0.8])
            )
            self._q0 = np.array(
                [6.2e-01, 1.7e00, 1.5e00, 6.9e-01, -1.3e00, 1.1e00, 1.5e-01]
            )
            OBSTACLE_HEIGHT = 0.85
            OBSTACLE_X = 2.0e-0
            OBSTACLE_Y = 0.5e-2
            OBSTACLE_Z = 0.5
            obstacles = [
                (
                    "obstacle1",
                    hppfcl.Box(OBSTACLE_X, OBSTACLE_Y, OBSTACLE_Z),
                    pin.SE3(
                        pin.utils.rotate("y", np.pi / 2),
                        np.array([-0.0, -0.1, OBSTACLE_HEIGHT]),
                    ),
                ),
            ]

        else:
            raise NotImplementedError(f"The input {name_scene} is not implemented.")

        # Adding all the obstacles to the geom model
        for obstacle in obstacles:
            name = obstacle[0]
            shape = obstacle[1]
            pose = obstacle[2]
            geom_obj = pin.GeometryObject(
                name,
                0,
                0,
                shape,
                pose,
            )
            self._cmodel.addGeometryObject(geom_obj)
        self._add_collision_pairs()
        return self._cmodel, self._target, self._q0

    def _add_collision_pairs(self):
        """Add the collision pairs in the collision model w.r.t to the chosen scene."""
        if self._name_scene == "box":
            self.obstacles = [
                "support_link_0",
                "obstacle1",
                "obstacle2",
                "obstacle3",
                "obstacle4",
            ]
            self.shapes_avoiding_collision = [
                "panda2_link7_capsule_0",
                "panda2_link7_capsule_1",
                "panda2_link6_capsule_0",
                "panda2_link5_capsule_1",
                "panda2_link5_capsule_0",
                "panda2_rightfinger_0",
                "panda2_leftfinger_0",
            ]
        elif self._name_scene == "ball":
            self.obstacles = ["obstacle1"]
            self.shapes_avoiding_collision = [
                "support_link_0",
                "panda2_leftfinger_0",
                "panda2_rightfinger_0",
                "panda2_link6_capsule_0",
                "panda2_link5_capsule_0",
                "panda2_link5_capsule_1",
            ]
        elif self._name_scene == "wall":
            self.obstacles = [
                "support_link_0",
                "obstacle1",
            ]
            self.shapes_avoiding_collision = [
                "panda2_link7_capsule_0",
                "panda2_link7_capsule_1",
                "panda2_link6_capsule_0",
                "panda2_link5_capsule_1",
                "panda2_link5_capsule_0",
                "panda2_rightfinger_0",
                "panda2_leftfinger_0",
            ]
        for shape in self.shapes_avoiding_collision:
            # Highlight the shapes of the robot that are supposed to avoid collision
            self._cmodel.geometryObjects[
                self._cmodel.getGeometryId(shape)
            ].meshColor = YELLOW_FULL
            for obstacle in self.obstacles:
                self._cmodel.addCollisionPair(
                    pin.CollisionPair(
                        self._cmodel.getGeometryId(shape),
                        self._cmodel.getGeometryId(obstacle),
                    )
                )

    def _add_collision_pairs_urdf(self):
        """Add the collision pairs in the collision model w.r.t to the chosen scene."""
        if self._name_scene == "box":
            self.shapes_avoiding_collision = [
                "panda2_link7_capsule_0",
                "panda2_link7_capsule_1",
                "panda2_link6_capsule_0",
                "panda2_link5_capsule_1",
                "panda2_link5_capsule_0",
                "panda2_rightfinger_0",
                "panda2_leftfinger_0",
            ]
        for shape in self.shapes_avoiding_collision:
            # Highlight the shapes of the robot that are supposed to avoid collision
            self._cmodel.geometryObjects[
                self._cmodel.getGeometryId(shape)
            ].meshColor = YELLOW_FULL
            for obstacle in self._obstacles_name:
                self._cmodel.addCollisionPair(
                    pin.CollisionPair(
                        self._cmodel.getGeometryId(shape),
                        self._cmodel.getGeometryId(obstacle),
                    )
                )

    def load_obstacle_urdf(self, urdf_filename: str):
        """Load models for a given URDF in the obstacle directory.

        Args:
            urdf_file_name (str): name of the URDF.
        """

        model_dir = dirname(dirname(((str(abspath(__file__))))))
        model_path = join(model_dir, "robot_description")
        urdf_dir = join(model_path, "urdf")
        obstacle_dir = join(urdf_dir, "obstacles")
        urdf_model_path = join(obstacle_dir, urdf_filename)

        model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_model_path)

        # changing the names of the frames because there is conflict between frames names of both models.
        for frame in model.frames:
            frame.name = frame.name + "_obstacle"

        self._obstacles_name = []
        for obstacle in collision_model.geometryObjects:
            self._obstacles_name.append(obstacle.name)

        return model, collision_model, visual_model


if __name__ == "__main__":
    from wrapper_meshcat import MeshcatWrapper
    from wrapper_panda import PandaWrapper

    # Creating the robot
    robot_wrapper = PandaWrapper(capsule=False)
    rmodel, cmodel, vmodel = robot_wrapper()

    scene = Scene()
    cmodel, target, q0 = scene.create_scene(rmodel, cmodel, "wall")

    rdata = rmodel.createData()
    cdata = cmodel.createData()
    # Generating the meshcat visualizer
    MeshcatVis = MeshcatWrapper()
    vis = MeshcatVis.visualize(
        robot_model=rmodel,
        robot_visual_model=cmodel,
        robot_collision_model=cmodel,
        TARGET=target,
    )
    vis[0].display(q0)

    pin.computeCollisions(rmodel, rdata, cmodel, cdata, pin.neutral(rmodel), False)
    for k in range(len(cmodel.collisionPairs)):
        cr = cdata.collisionResults[k]
        cp = cmodel.collisionPairs[k]
        print(
            "collision pair:",
            cp.first,
            ",",
            cp.second,
            "- collision:",
            "Yes" if cr.isCollision() else "No",
        )
