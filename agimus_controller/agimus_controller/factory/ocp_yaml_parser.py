import crocoddyl
import force_feedback_mpc
import numpy as np
import numpy.typing as npt
import pathlib
import pinocchio
import colmpc
import dataclasses
import typing as T

from agimus_controller.trajectory import (
    WeightedTrajectoryPoint,
)

# Always point this file. Not the file that calls get_default_yaml_file.
CURRENT_DIR = pathlib.Path(__file__).resolve().parent.parent


def get_default_yaml_file(basename: str) -> pathlib.Path:
    file = CURRENT_DIR / "ocp" / basename
    assert file.exists(), f"Default OCP YAML file '{file}' does not exist!"
    return file


def add_modules(values: dict):
    """Extends globals of this file with other custom globals. Enables importing of
    OCP components within `create_croco_dataclasses` from outside of this file.

    Args:
        values (dict): Dictionary with new globals to add.
    """
    globals().update(values)


def create_nested_dataclass(cls, values):
    kwargs = {k: create_croco_dataclasses(v) for k, v in values.items()}
    if hasattr(cls, "from_dict"):
        return cls.from_dict(kwargs)
    else:
        return cls(**kwargs)


def create_croco_dataclasses(values):
    if isinstance(values, dict) and "class" in values:
        cls = globals()[values["class"]]
        v = values.copy()
        v.pop("class")
        assert dataclasses.is_dataclass(cls), f"Class '{cls}' is not a dataclass!"
        return create_nested_dataclass(cls, v)
    elif isinstance(values, (list, tuple)):
        return type(values)(create_croco_dataclasses(v) for v in values)
    elif isinstance(values, dict):
        return dict((k, create_croco_dataclasses(v)) for k, v in values.items())
    else:
        return values


def as_dict(obj):
    if dataclasses.is_dataclass(obj):
        d = {
            field.name: as_dict(getattr(obj, field.name))
            for field in dataclasses.fields(obj)
        }
        d["class"] = obj.class_
        return d
    elif isinstance(obj, (list, tuple)):
        return type(obj)(as_dict(v) for v in obj)
    else:
        return obj


def get_frame_id(state: crocoddyl.StateMultibody, id: T.Union[str, int]) -> int:
    rmodel: pinocchio.Model = state.pinocchio
    if isinstance(id, str):
        assert rmodel.existFrame(id), f"Frame '{id}' does not exist!"
        id = rmodel.getFrameId(id)
    assert isinstance(id, int) and id < rmodel.nframes
    return id


@dataclasses.dataclass
class BuildData:
    state: crocoddyl.StateMultibody
    actuation: crocoddyl.ActuationModelAbstract
    collision_model: T.Optional[pinocchio.GeometryModel] = None
    # Transforms requested by the OCP and that needs to be provided
    # externally, typically using TF2 when using ROS. The keys are created
    # at build time. The values should be set before updating the OCP references.
    transforms: T.Dict[T.Tuple[str, str], T.Optional[pinocchio.SE3]] = (
        dataclasses.field(default_factory=dict)
    )


@dataclasses.dataclass
class ActivationModel:
    pass


@dataclasses.dataclass
class ActivationModelWeightedQuad(ActivationModel):
    class_: T.ClassVar[str] = "ActivationModelWeightedQuad"
    weights: T.Union[None, float, npt.NDArray[np.float64]] = None

    def update(self, data, obj, weights):
        obj.weights = weights

    def build(self, data: BuildData, residual: crocoddyl.CostModelResidual):
        if self.weights is None:
            weights = np.ones(residual.nr)
        else:
            try:
                weights = float(self.weights) * np.ones(residual.nr)
            except (ValueError, TypeError):
                weights = np.asarray(self.weights)
                assert weights.size == residual.nr
        return crocoddyl.ActivationModelWeightedQuad(weights)


@dataclasses.dataclass
class ActivationModelExp(ActivationModel):
    class_: T.ClassVar[str] = "ActivationModelExp"
    alpha: float = 1.0
    exponent: int = 1

    def build(self, data: BuildData, residual: crocoddyl.CostModelResidual):
        assert self.exponent in [1, 2]
        cls = (
            colmpc.ActivationModelExp
            if self.exponent == 1
            else colmpc.ActivationModelQuadExp
        )
        # float() is required to allow parsing a float in scientific notation.
        return cls(residual.nr, float(self.alpha))


# For backward compatibility
@dataclasses.dataclass
class ActivationModelQuadExp(ActivationModelExp):
    class_: T.ClassVar[str] = "ActivationModelQuadExp"
    exponent: int = 2

    def __post_init__(self):
        assert self.exponent == 2, (
            "ActivationModelQuadExp is provided for backward compatibility. exponent should be 2 (the default)."
        )


@dataclasses.dataclass
class ResidualModel:
    @staticmethod
    def needs_colmpc_freefwd_dynamics() -> bool:
        return False


@dataclasses.dataclass
class ResidualModelState(ResidualModel):
    class_: T.ClassVar[str] = "ResidualModelState"
    xref: T.Optional[npt.NDArray[np.float64]] = None

    def update(self, data, obj, pt: WeightedTrajectoryPoint):
        obj.reference = pt.point.robot_state
        return pt.weights.w_robot_state

    def build(self, data: BuildData):
        if self.xref is None:
            return crocoddyl.ResidualModelState(data.state)
        else:
            return crocoddyl.ResidualModelState(data.state, np.asarray(self.xref))


@dataclasses.dataclass
class ResidualModelControl(ResidualModel):
    class_: T.ClassVar[str] = "ResidualModelControl"
    uref: T.Optional[npt.NDArray[np.float64]] = None

    def update(self, data, obj, pt: WeightedTrajectoryPoint):
        obj.reference = pt.point.robot_effort
        return pt.weights.w_robot_effort

    def build(self, data: BuildData):
        if self.uref is None:
            return crocoddyl.ResidualModelControl(data.state)
        else:
            return crocoddyl.ResidualModelControl(data.state, np.asarray(self.uref))


@dataclasses.dataclass
class ResidualModelControlGrav(ResidualModel):
    class_: T.ClassVar[str] = "ResidualModelControlGrav"

    def update(self, data, obj, pt: WeightedTrajectoryPoint):
        obj.reference = pt.point.robot_effort
        return pt.weights.w_robot_effort

    def build(self, data: BuildData):
        return crocoddyl.ResidualModelControlGrav(data.state)


@dataclasses.dataclass
class ResidualModelFramePlacement(ResidualModel):
    class_: T.ClassVar[str] = "ResidualModelFramePlacement"
    id: T.Union[str, int]
    pref: T.Optional[npt.NDArray[np.float64]] = None

    def update(self, data, obj, pt: WeightedTrajectoryPoint):
        assert len(pt.point.end_effector_poses) == 1
        ee_name, ee_pose = next(iter(pt.point.end_effector_poses.items()))
        obj.id = get_frame_id(data.state, ee_name)
        obj.reference = ee_pose
        return pt.weights.w_end_effector_poses[ee_name]

    def build(self, data: BuildData):
        id = get_frame_id(data.state, self.id)
        if self.pref is None:
            pref = pinocchio.SE3.Identity()
        else:
            assert self.pref
            pref = pinocchio.XYZQUATToSE3(self.pref)
        return crocoddyl.ResidualModelFramePlacement(data.state, id, pref)


@dataclasses.dataclass
class ResidualModelFramePlacementStatic(ResidualModel):
    """Variant of ResidualModelFramePlacement requiring statically
    defined frame, for multi end effector support"""

    class_: T.ClassVar[str] = "ResidualModelFramePlacement"
    frame_id: T.Optional[str]
    pref: T.Optional[npt.NDArray[np.float64]] = None

    def update(self, data, obj, pt: WeightedTrajectoryPoint):
        assert self.frame_id in pt.point.end_effector_poses, (
            f"end_effector_poses should contains key {self.frame_id}"
        )
        obj.reference = pt.point.end_effector_poses[self.frame_id]
        return pt.weights.w_end_effector_poses[self.frame_id]

    def build(self, data: BuildData):
        id = get_frame_id(data.state, self.frame_id)
        if self.pref is None:
            pref = pinocchio.SE3.Identity()
        else:
            assert self.pref
            pref = pinocchio.XYZQUATToSE3(self.pref)
        return crocoddyl.ResidualModelFramePlacement(data.state, id, pref)


@dataclasses.dataclass
class ResidualModelFrameTranslation(ResidualModel):
    class_: T.ClassVar[str] = "ResidualModelFrameTranslation"
    id: T.Union[str, int]
    pref: T.Optional[npt.NDArray[np.float64]] = None

    def update(self, data, obj, pt: WeightedTrajectoryPoint):
        assert len(pt.point.end_effector_poses) == 1
        ee_name, ee_pose = next(iter(pt.point.end_effector_poses.items()))
        obj.id = get_frame_id(data.state, ee_name)
        obj.reference = ee_pose.translation
        return pt.weights.w_end_effector_poses[ee_name][:3]

    def build(self, data: BuildData):
        id = get_frame_id(data.state, self.id)
        if self.pref is None:
            pref = np.zeros(3)
        else:
            assert self.pref
            pref = self.pref[:3]
        return crocoddyl.ResidualModelFrameTranslation(data.state, id, pref)


@dataclasses.dataclass
class ResidualModelFrameTranslationStatic(ResidualModel):
    """Variant of ResidualModelFrameTranslation requiring statically
    defined frame, for multi end effector support"""

    class_: T.ClassVar[str] = "ResidualModelFrameTranslation"
    frame_id: T.Optional[str]
    pref: T.Optional[npt.NDArray[np.float64]] = None

    def update(self, data, obj, pt: WeightedTrajectoryPoint):
        assert self.frame_id in pt.point.end_effector_poses, (
            f"end_effector_poses should contains key {self.frame_id}"
        )
        obj.reference = pt.point.end_effector_poses[self.frame_id].translation
        return pt.weights.w_end_effector_poses[self.frame_id][:3]

    def build(self, data: BuildData):
        id = get_frame_id(data.state, self.frame_id)
        if self.pref is None:
            pref = np.zeros(3)
        else:
            assert self.pref
            pref = self.pref[:3]
        return crocoddyl.ResidualModelFrameTranslation(data.state, id, pref)


@dataclasses.dataclass
class ResidualModelFrameRotation(ResidualModel):
    class_: T.ClassVar[str] = "ResidualModelFrameRotation"
    id: T.Union[str, int]
    pref: T.Optional[npt.NDArray[np.float64]] = None

    def update(self, data, obj, pt: WeightedTrajectoryPoint):
        assert len(pt.point.end_effector_poses) == 1
        ee_name, ee_pose = next(iter(pt.point.end_effector_poses.items()))
        obj.id = get_frame_id(data.state, ee_name)
        obj.reference = ee_pose.rotation
        return pt.weights.w_end_effector_poses[ee_name][3:]

    def build(self, data: BuildData):
        id = get_frame_id(data.state, self.id)
        if self.pref is None:
            pref = np.eye(3)
        else:
            assert self.pref
            pref = pinocchio.Quaternion(self.pref[3:]).toRotationMatrix()
        return crocoddyl.ResidualModelFrameRotation(data.state, id, pref)


@dataclasses.dataclass
class ResidualModelFrameRotationStatic(ResidualModel):
    """Variant of ResidualModelFrameRotation requiring statically
    defined frame, for multi end effector support"""

    class_: T.ClassVar[str] = "ResidualModelFrameRotation"
    frame_id: T.Optional[str]
    pref: T.Optional[npt.NDArray[np.float64]] = None

    def update(self, data, obj, pt: WeightedTrajectoryPoint):
        assert self.frame_id in pt.point.end_effector_poses, (
            f"end_effector_poses should contains key {self.frame_id}"
        )
        obj.reference = pt.point.end_effector_poses[self.frame_id].rotation
        return pt.weights.w_end_effector_poses[self.frame_id][3:]

    def build(self, data: BuildData):
        id = get_frame_id(data.state, self.frame_id)
        if self.pref is None:
            pref = np.eye(3)
        else:
            assert self.pref
            pref = pinocchio.Quaternion(self.pref[3:]).toRotationMatrix()
        return crocoddyl.ResidualModelFrameRotation(data.state, id, pref)


@dataclasses.dataclass
class ResidualModelFrameVelocity(ResidualModel):
    class_: T.ClassVar[str] = "ResidualModelFrameVelocity"
    id: T.Union[str, int]
    pref: T.Optional[npt.NDArray[np.float64]] = None
    reference_frame: T.Optional[str] = "WORLD"

    def __post_init__(self):
        assert self.reference_frame in [
            "WORLD",
            "LOCAL",
            "LOCAL_WORLD_ALIGNED",
        ], (
            "ResidualModelFrameVelocity.reference_frame has to be one of: 'WORLD', 'LOCAL', 'LOCAL_WORLD_ALIGNED'."
        )

    def update(self, data, obj, pt: WeightedTrajectoryPoint):
        assert len(pt.point.end_effector_velocities) == 1
        ee_name, ee_vel = next(iter(pt.point.end_effector_velocities.items()))
        obj.id = get_frame_id(data.state, ee_name)
        obj.reference = ee_vel
        return pt.weights.w_end_effector_velocities[ee_name]

    def build(self, data: BuildData):
        id = get_frame_id(data.state, self.id)
        if self.pref is None:
            pref = pinocchio.Motion(np.zeros(6))
        else:
            assert self.pref
            pref = pinocchio.Motion(self.pref)
        frame = getattr(pinocchio, self.reference_frame)
        return crocoddyl.ResidualModelFrameVelocity(data.state, id, pref, frame)


@dataclasses.dataclass
class ResidualModelFrameVelocityStatic(ResidualModel):
    """Variant of ResidualModelFrameVelocity requiring statically
    defined frame, for multi end effector support"""

    class_: T.ClassVar[str] = "ResidualModelFrameVelocity"
    frame_id: T.Optional[str]
    pref: T.Optional[npt.NDArray[np.float64]] = None
    reference_frame: T.Optional[str] = "WORLD"

    def __post_init__(self):
        assert self.reference_frame in [
            "WORLD",
            "LOCAL",
            "LOCAL_WORLD_ALIGNED",
        ], (
            "ResidualModelFrameVelocity.reference_frame has to be one of: 'WORLD', 'LOCAL', 'LOCAL_WORLD_ALIGNED'."
        )

    def update(self, data, obj, pt: WeightedTrajectoryPoint):
        assert self.frame_id in pt.point.end_effector_velocities, (
            f"end_effector_velocities should contains key {self.frame_id}"
        )
        obj.reference = pt.point.end_effector_velocities[self.frame_id]
        return pt.weights.w_end_effector_poses[self.frame_id]

    def build(self, data: BuildData):
        id = get_frame_id(data.state, self.frame_id)
        if self.pref is None:
            pref = pinocchio.Motion(np.zeros(6))
        else:
            assert self.pref
            pref = pinocchio.Motion(self.pref)
        frame = getattr(pinocchio, self.reference_frame)
        return crocoddyl.ResidualModelFrameVelocity(data.state, id, pref, frame)


@dataclasses.dataclass
class ResidualModelVisualServoing(ResidualModel):
    """Visual servoing calculated as:
    wMf_target = wMo_vision * oMf_target
    where:
    - wMf_target is the target pose of the frame in the world frame
    - wMo_vision is the pose of an object frame in the world frame
    - oMf_target is the target pose of the frame in the object frame

    The frame f above is referred to as the robot frame below. It is the
    frame that will be controlled by MPC. It can be an end-effector frame,
    a camera frame (if the goal is to keep an object in the field of view)...
    """

    class_: T.ClassVar[str] = "ResidualModelVisualServoing"
    world_frame: str
    object_frame: str
    robot_frame: str

    def update(self, data: BuildData, obj, pt: WeightedTrajectoryPoint):
        assert self.input_key in pt.point.end_effector_poses, (
            f"end_effector_poses should contains key {self.input_key}"
        )

        weights = pt.weights.w_end_effector_poses[self.input_key]
        active = any(np.array(weights) != 0)

        wMo_vision = data.transforms[self.transforms_key]
        assert not active or wMo_vision is not None, (
            f"Weights are not all zeros and no transform for {self.transforms_key}"
        )

        oMf_target = pt.point.end_effector_poses[self.input_key]

        if wMo_vision is None:
            obj.reference = oMf_target
        else:
            obj.reference = wMo_vision * oMf_target
        return weights

    def build(self, data: BuildData):
        world_frame_id = get_frame_id(data.state, self.world_frame)
        f = data.state.pinocchio.frames[world_frame_id]
        assert f.parentJoint == 0, (
            f"Parent joint of world frame ({self.world_frame}) should be 0"
        )
        assert f.placement.isIdentity(), (
            f"Placement of world frame ({self.world_frame}) should be identity"
        )

        self.transforms_key = (self.world_frame, self.object_frame)
        self.input_key = self.robot_frame + "_vs"

        data.transforms.setdefault(self.transforms_key, None)

        frame_id = get_frame_id(data.state, self.robot_frame)
        pref = pinocchio.SE3.Identity()
        return crocoddyl.ResidualModelFramePlacement(data.state, frame_id, pref)


@dataclasses.dataclass
class ResidualDistanceCollisionBase(ResidualModel):
    # Ideally, this should go with an activation of type
    # ActivationModelExp with exponent 1.
    collision_pair: T.Tuple[str, str]

    def _collision_pair_id(self, cmodel: pinocchio.GeometryModel) -> int:
        assert len(self.collision_pair) == 2
        # Check that the collision pair exist in the collision model
        # and find its index in the list of collision pairs.
        cp = []
        for name in self.collision_pair:
            assert cmodel.existGeometryName(name), (
                f"Geometry object '{name}' not found."
            )
            cp.append(cmodel.getGeometryId(name))
        cp = pinocchio.CollisionPair(*cp)
        if not cmodel.existCollisionPair(cp):
            cmodel.addCollisionPair(cp)
        assert cmodel.existCollisionPair(cp)
        id = cmodel.findCollisionPair(cp)
        assert id < len(cmodel.collisionPairs)
        return id


@dataclasses.dataclass
class ResidualDistanceCollision(ResidualDistanceCollisionBase):
    class_: T.ClassVar[str] = "ResidualDistanceCollision"

    def build(self, data: BuildData):
        cmodel = data.collision_model
        id = self._collision_pair_id(cmodel)
        # Build the residual
        return colmpc.ResidualDistanceCollision(
            data.state, data.actuation.nu, cmodel, id
        )


@dataclasses.dataclass
class ResidualDistanceCollision2(ResidualDistanceCollisionBase):
    class_: T.ClassVar[str] = "ResidualDistanceCollision2"

    @staticmethod
    def needs_colmpc_freefwd_dynamics() -> bool:
        return True

    def build(self, data: BuildData):
        assert isinstance(data.state, colmpc.StateMultibody), (
            "The state should be of type colmpc.StateMultibody"
        )
        id = self._collision_pair_id(data.state.geometry)
        # Build the residual
        return colmpc.ResidualDistanceCollision2(data.state, data.actuation.nu, id)


@dataclasses.dataclass
class CostModel:
    residual: ResidualModel
    activation: T.Optional[ActivationModel] = None


@dataclasses.dataclass
class CostModelResidual(CostModel):
    class_: T.ClassVar[str] = "CostModelResidual"

    def build(self, data: BuildData):
        residual = self.residual.build(data)
        if self.activation is None:
            return crocoddyl.CostModelResidual(data.state, residual)
        else:
            activation = self.activation.build(data, residual)
            return crocoddyl.CostModelResidual(data.state, activation, residual)

    def update(self, data, obj, ref_w_pt: WeightedTrajectoryPoint):
        weights = self.residual.update(data, obj.residual, ref_w_pt)
        if self.activation is not None:
            self.activation.update(data, obj.activation, weights)


@dataclasses.dataclass
class CostModelSumItem:
    class_: T.ClassVar[str] = "CostModelSumItem"
    name: str
    cost: CostModel
    weight: float = 1.0
    active: bool = True
    update: bool = False
    publish_residual: bool = False


@dataclasses.dataclass
class ConstraintModel:
    residual: ResidualModel


@dataclasses.dataclass
class ConstraintModelResidual(ConstraintModel):
    class_: T.ClassVar[str] = "ConstraintModelResidual"
    lower: T.Optional[T.Union[float, npt.NDArray[np.float64]]] = None
    upper: T.Optional[T.Union[float, npt.NDArray[np.float64]]] = None
    active_on_terminal_node: bool = True

    def build(self, data: BuildData):
        residual = self.residual.build(data)
        if self.lower is None:
            lower = -np.inf * np.ones(residual.nr)
        else:
            try:
                lower = float(self.lower) * np.ones(residual.nr)
            except (ValueError, TypeError):
                lower = np.asarray(self.lower)
                assert lower.size == residual.nr
        if self.upper is None:
            upper = np.inf * np.ones(residual.nr)
        else:
            try:
                upper = float(self.upper) * np.ones(residual.nr)
            except (ValueError, TypeError):
                upper = np.asarray(self.upper)
                assert upper.size == residual.nr
        return crocoddyl.ConstraintModelResidual(
            data.state, residual, lower, upper, self.active_on_terminal_node
        )


@dataclasses.dataclass
class ConstraintModelControlLimit(ConstraintModelResidual):
    class_: T.ClassVar[str] = "ConstraintModelControlLimit"
    residual: T.Optional[ResidualModel] = dataclasses.field(default=None, init=False)
    lower: T.Optional[npt.NDArray[np.float64]] = dataclasses.field(
        default=None, init=False
    )
    upper: T.Optional[npt.NDArray[np.float64]] = dataclasses.field(
        default=None, init=False
    )

    def __post_init__(self):
        self.residual = ResidualModelControl()

    def build(self, data: BuildData):
        self.lower = -data.state.pinocchio.effortLimit
        self.upper = data.state.pinocchio.effortLimit
        return super().build(data)


@dataclasses.dataclass
class ConstraintListItem:
    name: str
    constraint: ConstraintModel
    active: bool = True


@dataclasses.dataclass
class DifferentialActionModel:
    pass


@dataclasses.dataclass
class DifferentialActionModelFreeFwdDynamics(DifferentialActionModel):
    class_: T.ClassVar[str] = "DifferentialActionModelFreeFwdDynamics"
    costs: T.List[CostModelSumItem]
    constraints: T.List[ConstraintListItem] = dataclasses.field(default_factory=list)

    @classmethod
    def from_dict(cls, kwargs: T.Dict[str, T.Any]):
        costs = [
            create_nested_dataclass(CostModelSumItem, v)
            for v in kwargs.get("costs", [])
        ]
        kwargs["costs"] = costs
        constraints = [
            create_nested_dataclass(ConstraintListItem, v)
            for v in kwargs.get("constraints", [])
        ]
        kwargs["constraints"] = constraints
        return DifferentialActionModelFreeFwdDynamics(**kwargs)

    def needs_colmpc_freefwd_dynamics(self) -> bool:
        for cost in self.costs:
            r = cost.cost.residual
            if r.needs_colmpc_freefwd_dynamics():
                return True
        if self.constraints:
            for constraint in self.constraints:
                r = constraint.constraint.residual
                if r.needs_colmpc_freefwd_dynamics():
                    return True
        return False

    def build(self, data: BuildData):
        costs = crocoddyl.CostModelSum(data.state)
        for cost in self.costs:
            c = cost.cost.build(data)
            costs.addCost(cost.name, c, cost.weight, cost.active)
        if self.needs_colmpc_freefwd_dynamics():
            DifferentialActionModelFreeFwdDynamics = (
                colmpc.DifferentialActionModelFreeFwdDynamics
            )
        else:
            DifferentialActionModelFreeFwdDynamics = (
                crocoddyl.DifferentialActionModelFreeFwdDynamics
            )
        if self.constraints is None:
            return DifferentialActionModelFreeFwdDynamics(
                data.state, data.actuation, costs
            )
        else:
            manager = crocoddyl.ConstraintModelManager(data.state)
            for constraint in self.constraints:
                c = constraint.constraint.build(data)
                manager.addConstraint(constraint.name, c, constraint.active)
            return DifferentialActionModelFreeFwdDynamics(
                data.state, data.actuation, costs, manager
            )

    def update(self, data, obj, pt: WeightedTrajectoryPoint):
        for cost in self.costs:
            if cost.update:
                # collision avoidance cost activation use no vectors of weights,
                # so we directly modify the scalar weight
                if isinstance(cost.cost.residual, ResidualDistanceCollisionBase):
                    obj.costs.costs[cost.name].weight = pt.weights.w_collision_avoidance
                else:
                    cost.cost.update(data, obj.costs.costs[cost.name].cost, pt)

        # At the moment, we do not need to add support for
        # updating the constraints


@dataclasses.dataclass
class IntegratedActionModelAbstract:
    differential: DifferentialActionModel
    step_time: float = 0.0
    with_cost_residual: bool = True

    def update(self, data, obj, pt: WeightedTrajectoryPoint):
        self.differential.update(data, obj.differential, pt)


@dataclasses.dataclass
class IntegratedActionModelEuler(IntegratedActionModelAbstract):
    class_: T.ClassVar[str] = "IntegratedActionModelEuler"

    def build(self, data: BuildData):
        differential = self.differential.build(data)
        return crocoddyl.IntegratedActionModelEuler(
            differential, self.step_time, self.with_cost_residual
        )


@dataclasses.dataclass
class DAMSoftContactAugmentedFwdDynamics(DifferentialActionModel):
    class_: T.ClassVar[str] = "DAMSoftContactAugmentedFwdDynamics"
    costs: T.List[CostModelSumItem]
    frame_id: str
    Kp: list[float]
    Kv: list[float]
    oPc: tuple[float, float, float]
    constraints: T.List[ConstraintListItem] = dataclasses.field(default_factory=list)
    with_gravity_torque_reg: bool = False
    enabled_directions: tuple[bool, bool, bool] = (True, True, True)
    ref: str = "LOCAL"
    cost_ref: str = "LOCAL"

    def __post_init__(self):
        self._dimension = sum(self.enabled_directions)
        assert self._dimension in [1, 3], "Soft contact is either 1D or 3D."

        for param in ("ref", "cost_ref"):
            assert getattr(self, param) in ["WORLD", "LOCAL", "LOCAL_WORLD_ALIGNED"], (
                f"DAMSoftContactAugmentedFwdDynamics.{param} has to be one of: "
                "'WORLD', 'LOCAL', 'LOCAL_WORLD_ALIGNED'."
            )

    @classmethod
    def from_dict(cls, kwargs: T.Dict[str, T.Any]):
        costs = [
            create_nested_dataclass(CostModelSumItem, v)
            for v in kwargs.get("costs", [])
        ]
        kwargs["costs"] = costs
        constraints = [
            create_nested_dataclass(ConstraintListItem, v)
            for v in kwargs.get("constraints", [])
        ]
        kwargs["constraints"] = constraints
        return DAMSoftContactAugmentedFwdDynamics(**kwargs)

    @property
    def _dam_cls_and_kwargs(self) -> tuple[type, dict]:
        if self._dimension == 1:
            axis = "xyz"[self.enabled_directions.index(1)]
            return force_feedback_mpc.DAMSoftContact1DAugmentedFwdDynamics, {
                "type": getattr(force_feedback_mpc.Vector3MaskType, axis)
            }
        if self._dimension == 3:
            return force_feedback_mpc.DAMSoftContact3DAugmentedFwdDynamics, {}
        raise ValueError("Soft contact is either 1D or 3D.")

    def needs_colmpc_freefwd_dynamics(self) -> bool:
        msg = "DAMSoftContactAugmentedFwdDynamics does not support colmpc Free Forward Dynamics!"
        for cost in self.costs:
            r = cost.cost.residual
            if r.needs_colmpc_freefwd_dynamics():
                raise ValueError(msg)
        if self.constraints:
            for constraint in self.constraints:
                r = constraint.constraint.residual
                if r.needs_colmpc_freefwd_dynamics():
                    raise ValueError(msg)
        return False

    def build(self, data: BuildData):
        costs = crocoddyl.CostModelSum(data.state)
        for cost in self.costs:
            c = cost.cost.build(data)
            costs.addCost(cost.name, c, cost.weight, cost.active)

        fid = data.state.pinocchio.getFrameId(self.frame_id)
        dam_cls, extra_kwargs = self._dam_cls_and_kwargs

        manager = crocoddyl.ConstraintModelManager(data.state)
        if self.constraints is not None:
            for constraint in self.constraints:
                c = constraint.constraint.build(data)
                manager.addConstraint(constraint.name, c, constraint.active)
            extra_kwargs.update({"constraints": manager})

        dam = dam_cls(
            state=data.state,
            actuation=data.actuation,
            costs=costs,
            frameId=fid,
            Kp=np.asarray(self.Kp),
            Kv=np.asarray(self.Kv),
            oPc=np.asarray(self.oPc),
            **extra_kwargs,
        )
        dam.with_gravity_torque_reg = self.with_gravity_torque_reg
        dam.tau_grav_weight = 0.0
        dam.with_force_cost = True
        dam.f_des = np.zeros(dam.nc)
        dam.f_weight = np.zeros(dam.nc)

        dam.ref = getattr(pinocchio, self.ref)
        dam.cost_ref_ = getattr(pinocchio, self.cost_ref)

        return dam

    def update(self, data, dam, pt: WeightedTrajectoryPoint):
        for cost in self.costs:
            if cost.update:
                # collision avoidance cost activation use no vectors of weights,
                # so we directly modify the scalar weight
                if isinstance(cost.cost.residual, ResidualDistanceCollisionBase):
                    dam.costs.costs[cost.name].weight = pt.weights.w_collision_avoidance
                else:
                    cost.cost.update(data, dam.costs.costs[cost.name].cost, pt)

        # Update the desired force.
        assert self.frame_id in pt.point.forces, (
            f"forces should contains key {self.frame_id}"
        )
        f_weight = pt.weights.w_forces[self.frame_id][:3]
        if np.sum(np.abs(f_weight)) > 1e-9:
            dam.active_contact = True
            dam.with_force_cost = True
            dam.f_des = pt.point.forces[self.frame_id].linear[self.enabled_directions]
            dam.f_weight = f_weight[self.enabled_directions]
        else:
            dam.active_contact = False
            dam.with_force_cost = False
            dam.f_des = np.zeros(dam.nc)
            dam.f_weight = np.zeros(dam.nc)

        if dam.with_gravity_torque_reg:
            dam.tau_grav_weight = pt.weights.w_robot_effort[0]


@dataclasses.dataclass
class IAMSoftContactAugmented(IntegratedActionModelAbstract):
    class_: T.ClassVar[str] = "IAMSoftContactAugmented"
    force_ub: T.Optional[npt.NDArray[np.float64]] = None
    force_lb: T.Optional[npt.NDArray[np.float64]] = None

    def __post_init__(self):
        if self.force_ub is not None:
            fub_len = len(self.force_ub)
            assert fub_len == 3 or fub_len == 1, (
                f"Incorrect size of force upper bound! Is {fub_len}, should be 1 or 3!"
            )

        if self.force_lb is not None:
            flb_len = len(self.force_lb)
            assert flb_len == 3 or flb_len == 1, (
                f"Incorrect size of force lower bound!, Is {flb_len}, should be 1 or 3!"
            )

    def update(self, data, obj, pt: WeightedTrajectoryPoint):
        self.differential.update(data, obj.differential, pt)

        if len(obj.differential.constraints.constraints) > 0:
            obj.g_lb = np.concatenate((obj.differential.g_lb, self.force_lb))
            obj.g_ub = np.concatenate((obj.differential.g_ub, self.force_ub))

    def build(self, data: BuildData):
        differential = self.differential.build(data)
        iam = force_feedback_mpc.IAMSoftContactAugmented(
            differential, self.step_time, self.with_cost_residual
        )
        # If no bounds are set, use the default ones
        force_dimension = sum(self.differential.enabled_directions)
        if self.force_ub is None:
            self.force_ub = iam.g_ub[-force_dimension:]
        else:
            assert len(self.force_ub) == force_dimension, (
                "Upper bound for forces does not match number "
                "of enabled force directions in Differential Action Model! "
                f"Value has {len(self.force_ub)} elements, while should have {force_dimension}"
            )
            self.force_ub = np.array(self.force_ub)

        if self.force_lb is None:
            self.force_lb = iam.g_lb[-force_dimension:]
        else:
            assert len(self.force_lb) == force_dimension, (
                "Lower bound for forces does not match number "
                "of enabled force directions in Differential Action Model!"
                f"Value has {len(self.force_lb)} elements, while should have {force_dimension}"
            )
            self.force_lb = np.array(self.force_lb)
        return iam


@dataclasses.dataclass
class ShootingProblem:
    running_model: IntegratedActionModelAbstract
    terminal_model: IntegratedActionModelAbstract

    def needs_colmpc_state(self) -> bool:
        return (
            self.running_model.differential.needs_colmpc_freefwd_dynamics()
            or self.terminal_model.differential.needs_colmpc_freefwd_dynamics()
        )

    def __post_init__(self):
        self.running_model = create_croco_dataclasses(self.running_model)
        self.terminal_model = create_croco_dataclasses(self.terminal_model)
