from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Union

import numpy as np
import numpy.typing as npt
import pinocchio as pin


@dataclass
class RobotModelParameters:
    q0_name = str()
    free_flyer = False
    locked_joint_names = []
    urdf = Path() | str
    srdf = Path() | str
    collision_as_capsule = False
    self_collision = False
    armature = npt.NDArray[np.float64]


class RobotModelFactory:
    """Parse the robot model, reduce it and filter the collision model."""

    """Complete model of the robot."""
    _complete_model = pin.Model()
    """ Complete model of the robot with collision meshes. """
    _complete_collision_model = pin.GeometryModel()
    """ Complete model of the robot with visualization meshes. """
    _complete_visual_model = pin.GeometryModel()
    """ Reduced model of the robot. """
    _rmodel = pin.Model()
    """ Reduced model of the robot with visualization meshes. """
    _rcmodel = pin.GeometryModel()
    """ Reduced model of the robot with collision meshes. """
    _rvmodel = pin.GeometryModel()
    """ Default configuration q0. """
    _q0 = np.array([])
    """ Parameters of the model. """
    _params = RobotModelParameters()
    """ Path to the collisions environment. """
    _env = Path()

    def load_model(self, param: RobotModelParameters, env: Union[Path, None]) -> None:
        self._params = param
        self._env = env
        self._load_pinocchio_models(param.urdf, param.free_flyer)
        self._load_default_configuration(param.srdf, param.q0_name)
        self._load_reduced_model(param.locked_joint_names, param.q0_name)
        self._update_collision_model(
            env, param.collision_as_capsule, param.self_collision, param.srdf
        )

    def _load_pinocchio_models(self, urdf: Path, free_flyer: bool) -> None:
        pass

    def _load_default_configuration(self, srdf_path: Path, q0_name: str) -> None:
        pass

    def _load_reduced_model(self, locked_joint_names, q0_name) -> None:
        pass

    def _update_collision_model(
        self,
        env: Union[Path, None],
        collision_as_capsule: bool,
        self_collision: bool,
        srdf: Path,
    ) -> None:
        pass

    def create_complete_robot_model(self) -> pin.Model:
        return self._complete_model.copy()

    def create_complete_collision_model(self) -> pin.GeometryModel:
        return self._complete_collision_model.copy()

    def create_complete_visual_model(self) -> pin.GeometryModel:
        return self._complete_visual_model.copy()

    def create_reduced_robot_model(self) -> pin.Model:
        return self._rmodel.copy()

    def create_reduced_collision_model(self) -> pin.GeometryModel:
        return self._rcmodel.copy()

    def create_reduced_visual_model(self) -> pin.GeometryModel:
        return self._rvmodel.copy()

    def create_default_configuration(self) -> np.array:
        return self._q0.copy()

    def create_model_parameters(self) -> RobotModelParameters:
        return deepcopy(self._params)

    def print_model(self):
        print("full model =\n", self._complete_model)
        print("reduced model =\n", self._rmodel)

    @property
    def armature(self) -> npt.NDArray[np.float64]:
        """Armature of the robot.

        Returns:
            npt.NDArray[np.float64]: Armature of the robot.
        """
        return self._params.armature
