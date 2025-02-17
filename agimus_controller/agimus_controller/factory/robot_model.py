from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Tuple

import coal
import numpy as np
import numpy.typing as npt
import pinocchio as pin


@dataclass
class RobotModelParameters:
    q0: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )  # Initial full configuration of the robot
    free_flyer: bool = False  # True if the robot has a free flyer
    moving_joint_names: list[str] = field(default_factory=list)
    robot_urdf: Union[Path, str] = (
        ""  # Path to the robot URDF file or string containing URDF as an XML
    )
    env_urdf: Union[None, Path, str] = (
        None  # Path to the environment URDF file or string containing URDF as an XML
    )
    srdf: Path = Path()  # Path to the SRDF file
    urdf_meshes_dir: Optional[Path] = (
        None  # Path to the directory containing the meshes and the URDF file.
    )
    collision_as_capsule: bool = (
        False  # True if the collision model should be reduced to capsules.
    )
    # By default, the collision model when convexified is a sum of spheres and cylinders, often representing capsules. Here, all the couples sphere cylinder sphere are replaced by coal capsules.
    self_collision: bool = False  # If True, the collision model takes into account collisions pairs written in the srdf file.
    armature: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )  # Default empty NumPy array
    collision_color: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([249.0, 136.0, 126.0, 125.0]) / 255.0
    )  # Red color for the collision model
    collision_pairs: list[Tuple[str, str]] = field(
        default_factory=list
    )  # list of collision pairs

    def __post_init__(self):
        # Handle armature:
        if self.armature.size == 0:
            # Use a default armature filled with 0s,
            # based on the size of moving_joint_names
            self.armature = np.zeros(len(self.moving_joint_names), dtype=np.float64)

        # Ensure armature has the same shape as moving_joint_names
        if (
            len(self.armature) != len(self.moving_joint_names) and not self.free_flyer
        ):  #! TODO: Do the same for free flyer
            raise ValueError(
                f"Armature must have the same shape as moving_joint_names. "
                f"Got {self.armature.shape} and {len(self.moving_joint_names)}."
            )

        # Ensure URDF and SRDF are valid
        if not self.robot_urdf:
            raise ValueError("Robot URDF can not be an empty string.")
        elif isinstance(self.robot_urdf, Path) and not self.robot_urdf.is_file():
            raise ValueError(
                f"Robot URDF must be a valid file path. File: '{self.robot_urdf}' doesn't exist!"
            )

        if isinstance(self.env_urdf, Path) and not self.env_urdf.is_file():
            raise ValueError(
                f"Environment URDF must be a valid file path. File: '{self.env_urdf}' doesn't exist!"
            )

        if not self.srdf.is_file():
            raise ValueError(
                f"SRDF must be a valid file path. File: '{self.srdf}' doesn't exist!"
            )

        if self.urdf_meshes_dir is not None and not self.urdf_meshes_dir.exists():
            raise ValueError(
                "urdf_meshes_dir must be a valid folder path. "
                f"Folder: '{self.urdf_meshes_dir}' doesn't exist!"
            )


class RobotModels:
    """Parse the robot model, reduce it and filter the collision model."""

    def __init__(self, param: RobotModelParameters):
        """Parse the robot model, reduce it and filter the collision model.

        Args:
            param (RobotModelParameters): Parameters to load the robot models.
        """
        self._params = param
        self._full_robot_model = None
        self._robot_model = None
        self._collision_model = None
        self._visual_model = None
        self.load_models()  # Populate models

    @property
    def params(self) -> RobotModelParameters:
        return self._params

    @property
    def full_robot_model(self) -> pin.Model:
        """Full robot model."""
        if self._full_robot_model is None:
            raise AttributeError("Full robot model has not been initialized yet.")
        return self._full_robot_model

    @property
    def robot_model(self) -> pin.Model:
        """Robot model, reduced if specified in the parameters."""
        if self._robot_model is None:
            raise AttributeError("Robot model has not been computed yet.")
        return self._robot_model

    @property
    def visual_model(self) -> pin.GeometryModel:
        """Visual model of the robot."""
        if self._visual_model is None:
            raise AttributeError("Visual model has not been computed yet.")
        return self._visual_model

    @property
    def collision_model(self) -> pin.GeometryModel:
        """Collision model of the robot."""
        if self._collision_model is None:
            raise AttributeError("Colision model has not been computed yet.")
        return self._collision_model

    def load_models(self) -> None:
        """Load and prepare robot models based on parameters."""
        self._q0 = deepcopy(self._params.q0)
        self._load_full_pinocchio_models()

        self._lock_joints()

        if self._params.collision_as_capsule:
            self._update_collision_model_to_capsules()
        if self._params.self_collision:
            self._update_collision_model_to_self_collision()
        self._add_collision_pairs()

    def _load_urdf(
        self,
        urdf: Path | str,
        use_free_flyer: bool,
        geometry_types: list[pin.GeometryType],
    ) -> tuple[pin.Model, pin.CollisionGeometry]:
        """Build pinocchio's models from URDF."""
        try:
            # load robot models
            if isinstance(urdf, Path):
                with open(urdf, "r") as file:
                    urdf = file.read().replace("\n", "")
            else:
                urdf = urdf
            if use_free_flyer:
                robot_model = pin.buildModelFromXML(urdf, pin.JointModelFreeFlyer())
            else:
                robot_model = pin.buildModelFromXML(urdf)
            package_dirs = (
                self._params.urdf_meshes_dir.absolute().as_posix()
                if self._params.urdf_meshes_dir is not None
                else None
            )
            geometry_type_models = [
                (
                    pin.buildGeomFromUrdfString(
                        robot_model,
                        urdf,
                        geometry_type,
                        package_dirs=package_dirs,
                    )
                )
                for geometry_type in geometry_types
            ]
        except Exception as e:
            raise ValueError(f"Failed to load URDF models from {urdf}: {e}")
        return (robot_model, *geometry_type_models)

    def _load_full_pinocchio_models(self) -> None:
        """Load the full robot model, the visual model and the collision model."""
        geometry_types = [
            pin.GeometryType.COLLISION,
            pin.GeometryType.VISUAL,
        ]
        self._full_robot_model, self._collision_model, self._visual_model = (
            self._load_urdf(
                self._params.robot_urdf, self._params.free_flyer, geometry_types
            )
        )
        if self._params.env_urdf is not None:
            env_model, env_collision_model, env_visual_model = self._load_urdf(
                self._params.env_urdf,
                use_free_flyer=False,
                geometry_types=geometry_types,
            )

            # make robot models append environment models
            self._full_robot_model, self._collision_model = pin.appendModel(
                self._full_robot_model,
                env_model,
                self._collision_model,
                env_collision_model,
                0,
                pin.SE3.Identity(),
            )
            _, self._visual_model = pin.appendModel(
                self._full_robot_model,
                pin.Model(),
                self.visual_model,
                env_visual_model,
                0,
                pin.SE3.Identity(),
            )

    def _lock_joints(self) -> None:
        """Apply locked joints."""
        # Sanity check.
        for jn in self._params.moving_joint_names:
            if jn not in self._full_robot_model.names:
                raise ValueError(jn + " not in the model.")
        # Find the joints to lock.
        joints_to_lock = []
        for jn in self._full_robot_model.names:
            if jn == "universe":
                continue
            if jn not in self._params.moving_joint_names:
                joints_to_lock.append(self._full_robot_model.getJointId(jn))
        if len(self._q0) == 0:
            self._q0 = np.zeros(self._full_robot_model.nq)
        (
            self._robot_model,
            [
                self._collision_model,
                self._visual_model,
            ],
        ) = pin.buildReducedModel(
            self._full_robot_model,
            [self._collision_model, self._visual_model],
            joints_to_lock,
            self._q0,
        )

    def _update_collision_model_to_capsules(self) -> None:
        """Update the collision model to capsules."""
        list_names_capsules = []
        geom_objects = self._collision_model.geometryObjects.copy()
        # Iterate through geometry objects in the collision model
        for geom_object in geom_objects:
            geometry = geom_object.geometry
            # Convert cylinders to capsules
            if isinstance(geometry, coal.Cylinder):
                # Remove superfluous suffix from the name
                split_name = geom_object.name.split("_")
                base_name = "_".join(split_name[:-1])
                if sum(1 for obj in geom_objects if base_name in obj.name) < 3:
                    continue
                id = int(split_name[-1])
                name = self._generate_capsule_name(base_name, list_names_capsules)
                list_names_capsules.append(name)
                capsule = pin.GeometryObject(
                    name=name,
                    parent_frame=geom_object.parentFrame,
                    parent_joint=geom_object.parentJoint,
                    collision_geometry=coal.Capsule(
                        geometry.radius, geometry.halfLength
                    ),
                    placement=geom_object.placement,
                )
                capsule.meshColor = self._params.collision_color
                self._collision_model.addGeometryObject(capsule)
                self._collision_model.removeGeometryObject(geom_object.name)
                self._collision_model.removeGeometryObject(
                    base_name + "_" + str(id + 1)
                )
                self._collision_model.removeGeometryObject(
                    base_name + "_" + str(id + 2)
                )
            # Remove useless meshes.
            elif (
                not isinstance(geometry, coal.Sphere)
                and not isinstance(geometry, coal.Box)
                and not isinstance(geometry, coal.Cylinder)
            ):
                self._collision_model.removeGeometryObject(geom_object.name)

    def _update_collision_model_to_self_collision(self) -> None:
        """Update the collision model to self collision."""
        self._collision_model.addAllCollisionPairs()
        pin.removeCollisionPairs(
            self._robot_model,
            self._collision_model,
            str(self._params.srdf.absolute()),
        )

    def _add_collision_pairs(self) -> None:
        """Add collision pairs to the collision model."""

        for collision_pair in self._params.collision_pairs:
            try:
                geom1_name = collision_pair[0]
                geom2_name = collision_pair[1]
                geom1_id = self.collision_model.getGeometryId(geom1_name)
                geom2_id = self.collision_model.getGeometryId(geom2_name)
                self.collision_model.addCollisionPair(
                    pin.CollisionPair(geom1_id, geom2_id)
                )
            except Exception as e:
                raise ValueError(
                    f"Invalid collision pair with names {geom1_name} {geom2_name} got error : {e}"
                )

    def _generate_capsule_name(self, base_name: str, existing_names: list[str]) -> str:
        """Generates a unique capsule name for a geometry object.
        Args:
            base_name (str): The base name of the geometry object.
            existing_names (list): List of names already assigned to capsules.
        Returns:
            str: Unique capsule name.
        """
        i = 0
        while f"{base_name}_capsule_{i}" in existing_names:
            i += 1
        return f"{base_name}_capsule_{i}"

    @property
    def armature(self) -> npt.NDArray[np.float64]:
        """Armature of the robot.
        Returns:
            npt.NDArray[np.float64]: Armature of the robot.
        """
        return self._params.armature
