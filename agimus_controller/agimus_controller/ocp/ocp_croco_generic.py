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
class ActivationModelQuadExp(ActivationModel):
    class_: T.ClassVar[str] = "ActivationModelQuadExp"
    alpha: float = 1.0

    def build(self, data: BuildData, residual: crocoddyl.CostModelResidual):
        # float() is required to allow parsing a float in scientific notation.
        return colmpc.ActivationModelQuadExp(residual.nr, float(self.alpha))


@dataclasses.dataclass
class ResidualModel:
    pass


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
class ResidualDistanceCollision(ResidualModel):
    class_: T.ClassVar[str] = "ResidualDistanceCollision"
    collision_pair: T.Tuple[str, str]

    def build(self, data: BuildData):
        assert len(self.collision_pair) == 2
        cmodel = data.collision_model
        # Check that the collision pair exist in the collision model
        # and find its index in the list of collision pairs.
        cp = []
        for name in self.collision_pair:
            assert cmodel.existGeometryName(name), f"Geometry object {name} not found."
            cp.append(cmodel.getGeometryId(name))
        cp = pinocchio.CollisionPair(*cp)
        assert cmodel.existCollisionPair(cp)
        id = cmodel.findCollisionPair(cp)
        assert id < len(data.collision_model.collisionPairs)
        # Build the residual
        return colmpc.ResidualDistanceCollision(
            data.state, data.actuation.nu, cmodel, id
        )


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

    def build(self, data: BuildData):
        costs = crocoddyl.CostModelSum(data.state)
        for cost in self.costs:
            c = cost.cost.build(data)
            costs.addCost(cost.name, c, cost.weight, cost.active)
        if self.constraints is None:
            return crocoddyl.DifferentialActionModelFreeFwdDynamics(
                data.state, data.actuation, costs
            )
        else:
            manager = crocoddyl.ConstraintModelManager(data.state)
            for constraint in self.constraints:
                c = constraint.constraint.build(data)
                manager.addConstraint(constraint.name, c, constraint.active)
            return crocoddyl.DifferentialActionModelFreeFwdDynamics(
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
        super().__init__(robot_models, params)
        self.init_debug_data_references_and_residuals()

    @property
    def _build_data(self) -> BuildData:
        if not hasattr(self, "_build_data_obj"):
            self._build_data_obj = BuildData(
                self._state, self._actuation, self._collision_model
            )
        return self._build_data_obj

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

    def fill_debug_data_references_and_residuals(self) -> None:
        """Fill references and residuals of dataclass OCPDebugData."""
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

    def solve(
        self,
        x0: npt.NDArray[np.float64],
        x_warmstart: list[npt.NDArray[np.float64]],
        u_warmstart: list[npt.NDArray[np.float64]],
        use_iteration_limits_and_timeout: bool = True,
    ) -> None:
        super().solve(
            x0=x0,
            x_warmstart=x_warmstart,
            u_warmstart=u_warmstart,
            use_iteration_limits_and_timeout=use_iteration_limits_and_timeout,
        )
        if self._ocp_params.use_debug_data:
            self.fill_debug_data_references_and_residuals()
