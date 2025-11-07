import crocoddyl
import numpy as np
import pinocchio
import yaml
import pinocchio as pin
import typing as T


from agimus_controller.factory.ocp_yaml_parser import (
    ShootingProblem,
    BuildData,
    ResidualDistanceCollisionBase,
)
from agimus_controller.ocp_base_croco import (
    OCPBaseCroco,
    RobotModels,
    OCPParamsBaseCroco,
)
from agimus_controller.trajectory import (
    WeightedTrajectoryPoint,
)


class OCPCrocoGeneric(OCPBaseCroco):
    def __init__(
        self,
        robot_models: RobotModels,
        params: OCPParamsBaseCroco,
        yaml_file: T.Union[str, T.IO],
        expect_rolling_buffer: bool = False,
    ) -> None:
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)
        self._data = ShootingProblem(**data)
        super().__init__(
            robot_models, params, use_colmpc_state=self._data.needs_colmpc_state()
        )
        self._expect_rolling_buffer = expect_rolling_buffer
        self._first_call = True
        self.init_debug_data_attributes()

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

    def init_debug_data_attributes(self) -> None:
        """
        Initialize references and residuals of dataclass OCPDebugData.
        """
        for cost in self._data.running_model.differential.costs:
            # collision avoidance costs only has changes in weights, not references.
            if cost.update and not isinstance(
                cost.cost.residual, ResidualDistanceCollisionBase
            ):
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
        if self._expect_rolling_buffer:
            if self._first_call:
                for running_model, ref_weighted_pt in zip(
                    problem.runningModels, reference_weighted_trajectory[:-1]
                ):
                    self._data.running_model.update(
                        self._build_data, running_model, ref_weighted_pt
                    )
                self._first_call = False
            else:
                problem.circularAppend(problem.runningModels[0])

            self._data.running_model.update(
                self._build_data,
                problem.runningModels[-1],
                reference_weighted_trajectory[-2],
            )
        else:
            for running_model, ref_weighted_pt in zip(
                problem.runningModels, reference_weighted_trajectory[:-1]
            ):
                self._data.running_model.update(
                    self._build_data, running_model, ref_weighted_pt
                )

        self._data.terminal_model.update(
            self._build_data, problem.terminalModel, reference_weighted_trajectory[-1]
        )
