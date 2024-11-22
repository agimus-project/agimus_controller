from pathlib import Path
import numpy as np
import pinocchio as pin


YELLOW_FULL = np.array([1, 1, 0, 1.0])


class Scene:
    def __init__(
        self,
        name_scene: str,
        q_init,
        obstacle_pose=None,
    ) -> None:
        """Create the scene that encapsulates the obstacles.

        Args:
            name_scene (str): Name of the scene, among "box", "ball" and "wall".
            obstacle_pose (pin.SE3, optional): Pose of the obstacles. The default one is adapted for each scene. Defaults to None.

        Raises:
            NotImplementedError: No scene of the given name.
        """

        self._name_scene = name_scene
        self.obstacle_pose = obstacle_pose
        self._q0 = q_init
        if self._name_scene == "box":
            self.urdf_filename = "box.urdf"
            self._TARGET_POSE1 = pin.SE3(
                pin.utils.rotate("x", np.pi), np.array([0, -0.4, 0.47])
            )
            self._TARGET_POSE2 = pin.SE3(
                pin.utils.rotate("x", np.pi), np.array([0, 0.15, 0.47])
            )
            if self.obstacle_pose is None:
                self.obstacle_pose = pin.SE3.Identity()
                self.obstacle_pose.translation = np.array([0, 0.15, 0.37])
        elif self._name_scene == "ball":
            self.urdf_filename = "ball.urdf"
            self._TARGET_POSE1 = pin.SE3(
                pin.utils.rotate("x", np.pi), np.array([0.475, -0.1655, 1.27])
            )
            self._TARGET_POSE2 = pin.SE3(
                pin.utils.rotate("x", np.pi), np.array([0, -0.4, 1.12])
            )
            if self.obstacle_pose is None:
                self.obstacle_pose = pin.SE3.Identity()
                self.obstacle_pose.translation = np.array([0.25, -0.4, 1.12])
        elif self._name_scene == "wall":
            self.urdf_filename = "wall.urdf"
            self._TARGET_POSE1 = pin.SE3(
                pin.utils.rotate("x", np.pi), np.array([0, -0.4, 0.47])
            )
            self._TARGET_POSE2 = pin.SE3(
                pin.utils.rotate("x", np.pi), np.array([0, 0.15, 0.47])
            )
            if self.obstacle_pose is None:
                self.obstacle_pose = pin.SE3.Identity()
                self.obstacle_pose.translation = np.array([0.6, 0.5, 1.0])
        else:
            raise NotImplementedError(
                f"The input {self._name_scene} is not implemented."
            )

    def create_scene_from_urdf(
        self,
        rmodel: pin.Model,
        cmodel: pin.Model,
    ):
        """Create a scene among the one described in the constructor of the class.

        Args:
            rmodel (pin.Model): robot model
            cmodel (pin.Model): collision model of the robot
        """
        obs_model, obs_cmodel, _ = self._load_obstacle_urdf(self.urdf_filename)
        self._rmodel, self._cmodel = pin.appendModel(
            rmodel,
            obs_model,
            cmodel,
            obs_cmodel,
            0,
            self.obstacle_pose,
        )
        self._add_collision_pairs_urdf()
        return (
            self._rmodel,
            self._cmodel,
            self._TARGET_POSE1,
            self._TARGET_POSE2,
            self._q0,
        )

    def _add_collision_pairs_urdf(self):
        """Add the collision pairs in the collision model w.r.t to the chosen scene."""
        self.get_shapes_avoiding_collision()
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
            # Add the collision pair with the support link 0 because this is the table on which sits the robot.
            self._cmodel.addCollisionPair(
                pin.CollisionPair(
                    self._cmodel.getGeometryId(shape),
                    self._cmodel.getGeometryId("support_link_0"),
                )
            )

    def _load_obstacle_urdf(self, urdf_filename: str):
        """Load models for a given URDF in the obstacle directory.

        Args:
            urdf_file_name (str): name of the URDF.
        """
        obstacle_dir = (
            Path(__file__).resolve().parent.parent / "resources" / "obstacles"
        )
        self.urdf_model_path = obstacle_dir / urdf_filename
        obs_model, obs_cmodel, obs_vmodel = pin.buildModelsFromUrdf(
            str(self.urdf_model_path)
        )

        # changing the names of the frames because there is conflict between frames names of both models.
        print("Frame names of the obstacle:")
        for frame in obs_model.frames:
            frame.name = frame.name + "_obstacle"
            print("\t- ", frame.name)
        self._obstacles_name = []
        for obstacle in obs_cmodel.geometryObjects:
            self._obstacles_name.append(obstacle.name)

        self._obstacles_name.append("support_link_0")
        return obs_model, obs_cmodel, obs_vmodel

    def get_shapes_avoiding_collision(self):
        """Get the list of the shapes avoiding the collisions with the obstacles.

        Returns:
            list: list of the shapes avoiding the collisions with the obstacles.
        """
        if self._name_scene == "box" or "wall":
            self.shapes_avoiding_collision = [
                "panda_link5_sc_capsule_1",
                "panda_leftfinger_0",
            ]
        elif self._name_scene == "ball":
            self.shapes_avoiding_collision = [
                "panda_link5_sc_capsule_1",
                "panda_leftfinger_0",
            ]
        else:
            raise NotImplementedError(
                f"The input {self._name_scene} is not implemented."
            )

        return self.shapes_avoiding_collision
