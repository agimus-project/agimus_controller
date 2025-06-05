import pathlib
import crocoddyl
import numpy as np
import numpy.typing as npt
import pinocchio
import colmpc
import dataclasses
import yaml
import pinocchio as pin
import typing as T

from agimus_controller.ocp_base_croco import (
    OCPBaseCroco,
    RobotModels,
    OCPParamsBaseCroco,
)
from agimus_controller.trajectory import (
    WeightedTrajectoryPoint,
)


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
        assert dataclasses.is_dataclass(cls)
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
            "ActivationModelQuadExp is provided for backward compatibility. exponent should not be 2 (the default)."
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
class ResidualModelFramePlacement(ResidualModel):
    class_: T.ClassVar[str] = "ResidualModelFramePlacement"
    id: T.Union[str, int]
    pref: T.Optional[npt.NDArray[np.float64]] = None

    def update(self, data, obj, pt: WeightedTrajectoryPoint):
        assert len(pt.point.end_effector_poses) == 1
        ee_name, ee_pose = next(iter(pt.point.end_effector_poses.items()))
        obj.id = self._get_id(data.state, ee_name)
        obj.reference = ee_pose
        return pt.weights.w_end_effector_poses[ee_name]

    @staticmethod
    def _get_id(state, id):
        rmodel: pinocchio.Model = state.pinocchio
        if isinstance(id, str):
            assert rmodel.existFrame(id)
            id = rmodel.getFrameId(id)
        assert isinstance(id, int) and id < rmodel.nframes
        return id

    def build(self, data: BuildData):
        id = self._get_id(data.state, self.id)
        if self.pref is None:
            pref = pinocchio.SE3.Identity()
        else:
            assert self.pref
            pref = pinocchio.XYZQUATToSE3(self.pref)
        return crocoddyl.ResidualModelFramePlacement(data.state, id, pref)


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

    @staticmethod
    def _get_id(state: crocoddyl.StateMultibody, name: str):
        rmodel: pinocchio.Model = state.pinocchio
        assert rmodel.existFrame(name), (
            f"{name} not found in model. Frames are {[f.name for f in rmodel.frames]}"
        )
        id = rmodel.getFrameId(name)
        assert isinstance(id, int) and id < rmodel.nframes
        return id

    def build(self, data: BuildData):
        world_frame_id = self._get_id(data.state, self.world_frame)
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

        frame_id = self._get_id(data.state, self.robot_frame)
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
            assert cmodel.existGeometryName(name), f"Geometry object {name} not found."
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


class OCPCrocoGeneric(OCPBaseCroco):
    def __init__(
        self,
        robot_models: RobotModels,
        params: OCPParamsBaseCroco,
        yaml_file: T.Union[str, T.IO],
    ) -> None:
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)
        self._data = ShootingProblem(**data)
        super().__init__(
            robot_models, params, use_colmpc_state=self._data.needs_colmpc_state()
        )
        self.init_debug_data_references_and_residuals()

    @property
    def _build_data(self) -> BuildData:
        if not hasattr(self, "_build_data_obj"):
            self._build_data_obj = BuildData(
                self._state, self._actuation, self._collision_model
            )
        return self._build_data_obj

    @property
    def input_transforms(self) -> T.Dict[T.Tuple[str, str], pinocchio.SE3]:
        """Returns a dictionary of transforms that the OCP needs as input.
        The keys are tuples of frame names. The first is the parent frame and
        the second is the child frame.
        The values are the current transform."""
        return self._build_data.transforms

    def create_running_model_list(self) -> list[crocoddyl.ActionModelAbstract]:
        running_model_list = []
        for dt in self._ocp_params.timesteps:
            running_model = self._data.running_model.build(self._build_data)
            running_model.differential.armature = self._robot_models.armature
            running_model.dt = dt
            running_model_list.append(running_model)
        assert len(running_model_list) == self.n_controls
        return running_model_list

    def create_terminal_model(self) -> crocoddyl.ActionModelAbstract:
        terminal_model = self._data.terminal_model.build(self._build_data)
        terminal_model.differential.armature = self._robot_models.armature
        terminal_model.dt = 0.0
        return terminal_model

    def init_debug_data_references_and_residuals(self) -> None:
        """Initialize references and residuals of dataclass OCPDebugData."""
        for cost in self._data.running_model.differential.costs:
            if cost.update:
                self._debug_data.references.append((cost.name, None))
            if cost.publish_residual:
                self._debug_data.residuals.append((cost.name, None))

    def fill_debug_data(self, res, ocp_results) -> None:
        super().fill_debug_data(res=res, ocp_results=ocp_results)
        model = self._problem.runningModels[0]
        # fill references data only for first node because when resetting ocp, we are
        # shifting reference to next node, so we end up publishing all the references
        for reference_idx in range(len(self._debug_data.references)):
            name = self._debug_data.references[reference_idx][0]
            reference = model.differential.costs.costs[name].cost.residual.reference
            if isinstance(reference, pin.SE3):
                reference = pin.SE3ToXYZQUAT(reference)
            self._debug_data.references[reference_idx] = (name, reference)

        # fill residuals data
        for residual_idx in range(len(self._debug_data.residuals)):
            name = self._debug_data.residuals[residual_idx][0]
            residual_size = (
                self._problem.runningDatas[0]
                .differential.costs.costs[name]
                .residual.r.shape[0]
            )
            residual_prediction = np.zeros((self._ocp_params.n_controls, residual_size))
            for node_idx, data in enumerate(self._problem.runningDatas):
                residual_prediction[node_idx, :] = data.differential.costs.costs[
                    name
                ].residual.r
            self._debug_data.residuals[residual_idx] = (name, residual_prediction)

    def set_reference_weighted_trajectory(
        self, reference_weighted_trajectory: list[WeightedTrajectoryPoint]
    ):
        """Set the reference trajectory for the OCP."""

        assert len(reference_weighted_trajectory) == self.n_controls + 1

        problem = self._solver.problem

        # Modify running costs reference and weights
        for running_model, ref_weighted_pt in zip(
            problem.runningModels, reference_weighted_trajectory[:-1]
        ):
            self._data.running_model.update(
                self._build_data, running_model, ref_weighted_pt
            )

        self._data.terminal_model.update(
            self._build_data, problem.terminalModel, reference_weighted_trajectory[-1]
        )

    @staticmethod
    def get_default_yaml_file(basename: str) -> pathlib.Path:
        file = pathlib.Path(__file__).parent / basename
        return file
