import dataclasses
import typing as T
import yaml

import crocoddyl
import force_feedback_mpc
import mim_solvers
import numpy as np
import numpy.typing as npt
import pinocchio
import resource_retriever as r
from agimus_controller.factory.robot_model import RobotModels
from agimus_controller.mpc_data import OCPDebugData, OCPResults
from agimus_controller.ocp.ocp_croco_generic import (
    BuildData,
    CostModelSumItem,
    ConstraintListItem,
    DifferentialActionModel,
    IntegratedActionModelAbstract,
    ResidualDistanceCollisionBase,
    OCPCrocoGeneric,
    ShootingProblem,
    create_nested_dataclass,
)
from agimus_controller.ocp_param_base import OCPParamsBaseCroco
from agimus_controller.trajectory import (
    WeightedTrajectoryPoint,
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
        assert force_feedback_mpc is not None, "Module force_feedback_mpc not found"

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
        assert force_feedback_mpc is not None, "Module force_feedback_mpc not found"

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
        # If None use default
        if self.force_ub is None:
            self.force_ub = iam.g_ub[-3:]
        else:
            self.force_ub = np.array(self.force_ub)

        if self.force_lb is None:
            self.force_lb = iam.g_lb[-3:]
        else:
            self.force_lb = np.array(self.force_lb)
        return iam


class OCPCrocoForceFeedbackGeneric(OCPCrocoGeneric):
    def __init__(
        self,
        robot_models: RobotModels,
        ocp_params: OCPParamsBaseCroco,
        yaml_file: T.Union[str, T.IO],
        expect_rolling_buffer: bool = False,
    ) -> None:
        data = yaml.safe_load(r.get(yaml_file))
        self._data = ShootingProblem(**data)
        self._enabled_directions = (
            self._data.running_model.differential.enabled_directions
        )
        self._expect_rolling_buffer = expect_rolling_buffer

        # Setting the robot model
        self._robot_models = robot_models
        self._collision_model = self._robot_models.collision_model
        self._armature = self._robot_models.armature

        # Stat and actuation model
        self._state = crocoddyl.StateMultibody(self._robot_models.robot_model)
        self._actuation = crocoddyl.ActuationModelFull(self._state)

        # Setting the OCP parameters
        self._ocp_params = ocp_params
        self._solver = None
        self._ocp_results: OCPResults = None
        self._debug_data: OCPDebugData = OCPDebugData()

        # Create the running models
        self._running_model_list = self.create_running_model_list()
        # Create the terminal model
        self._terminal_model = self.create_terminal_model()
        # Create the shooting problem
        self._problem = crocoddyl.ShootingProblem(
            np.zeros(
                self._robot_models.robot_model.nq
                + self._robot_models.robot_model.nv
                + sum(self._enabled_directions)
            ),
            self._running_model_list,
            self._terminal_model,
        )
        self._problem.nthreads = ocp_params.n_threads
        # Create solver + callbacks
        self._solver = mim_solvers.SolverCSQP(self._problem)

        # Merit function
        self._solver.use_filter_line_search = self._ocp_params.use_filter_line_search

        # Parameters of the solver
        if self._ocp_params.max_solve_time is not None:
            self._solver.max_solve_time = self._ocp_params.max_solve_time
        self._solver.termination_tolerance = self._ocp_params.termination_tolerance
        self._solver.max_qp_iters = self._ocp_params.qp_iters
        self._solver.eps_abs = self._ocp_params.eps_abs
        self._solver.eps_rel = self._ocp_params.eps_rel

        if self._ocp_params.callbacks:
            self._solver.setCallbacks(
                [mim_solvers.CallbackVerbose(), mim_solvers.CallbackLogger()]
            )

        self.init_debug_data_attributes()
        self._first_call = True

    @property
    def enabled_directions(self) -> int:
        return self._enabled_directions

    @property
    def frame_id(self) -> str:
        return self._data.running_model.differential.frame_id

    @property
    def oPc(self) -> npt.ArrayLike:
        return np.asarray(self._data.running_model.differential.oPc)


def get_globals():
    return globals()
