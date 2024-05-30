import numpy as np
import pinocchio as pin

import hppfcl

from wrapper_meshcat import YELLOW_FULL
class Scene:
    def __init__(self) -> None:
        pass

    def create_scene(self, rmodel: pin.Model, cmodel: pin.Model, name_scene: str):
        """Create a scene amond the ones : "box"

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
            self._cmodel.geometryObjects[self._cmodel.getGeometryId(shape)].meshColor = YELLOW_FULL
            for obstacle in self.obstacles:
                self._cmodel.addCollisionPair(
                    pin.CollisionPair(
                        self._cmodel.getGeometryId(shape),
                        self._cmodel.getGeometryId(obstacle),
                    )
                )


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
