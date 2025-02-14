import crocoddyl
import numpy as np
import pinocchio as pin
from typing import Tuple
from typing import Union
import numpy.typing as npt

from agimus_controller.ocp_base_croco import OCPBaseCroco
from agimus_controller.trajectory import WeightedTrajectoryPoint

from colmpc import ActivationModelQuadExp, ResidualDistanceCollision


class OCPCrocoTrajTrackCollAvoidance(OCPBaseCroco):
    def create_residuals(
        self,
    ) -> Tuple[
        crocoddyl.ResidualModelAbstract,
        crocoddyl.ResidualModelAbstract,
        crocoddyl.ResidualModelAbstract,
    ]:
        """Create state, control, and end-effector placement residuals."""
        x_residual = crocoddyl.ResidualModelState(
            self._state,
            np.concatenate(
                (
                    pin.neutral(self._robot_models.robot_model),
                    np.zeros(self._robot_models.robot_model.nv),
                )
            ),
        )
        u_residual = crocoddyl.ResidualModelControl(self._state)
        frame_placement_residual = crocoddyl.ResidualModelFramePlacement(
            self._state,
            0,
            pin.SE3.Identity(),
        )

        return x_residual, u_residual, frame_placement_residual

    def create_running_model_list(self) -> list[crocoddyl.ActionModelAbstract]:
        running_model_list = []
        running_cost_model = crocoddyl.CostModelSum(self._state)
        x_residual, u_residual, frame_placement_residual = self.create_residuals()
        x_reg_cost = crocoddyl.CostModelResidual(
            self._state,
            crocoddyl.ActivationModelWeightedQuad(
                np.ones(
                    self._robot_models.robot_model.nq
                    + self._robot_models.robot_model.nv
                )
            ),
            x_residual,
        )
        u_reg_cost = crocoddyl.CostModelResidual(
            self._state,
            crocoddyl.ActivationModelWeightedQuad(
                np.ones(self._robot_models.robot_model.nv)
            ),
            u_residual,
        )
        ee_pose_cost = crocoddyl.CostModelResidual(
            self._state,
            crocoddyl.ActivationModelWeightedQuad(np.ones(6)),
            frame_placement_residual,
        )
        for col_idx in range(len(self._collision_model.collisionPairs)):
            activation = ActivationModelQuadExp(
                1, self._params.activation_distance_threshold**2
            )
            distance_coll_residual = ResidualDistanceCollision(
                self._state, 7, self._collision_model, col_idx
            )
            distance_coll_cost = crocoddyl.CostModelResidual(
                self._state, activation, distance_coll_residual
            )
            running_cost_model.addCost(f"distColl{col_idx}", distance_coll_cost, 1.0)
        running_cost_model.addCost("stateReg", x_reg_cost, 1.0)
        running_cost_model.addCost("ctrlReg", u_reg_cost, 1.0)
        running_cost_model.addCost("goalTracking", ee_pose_cost, 1.0)

        constraints = self._get_collision_constraints(active=True)
        running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self._state, self._actuation, running_cost_model, constraints
        )
        for factor, dts in zip(
            self._params.dt_factor_n_seq.factors, self._params.dt_factor_n_seq.dts
        ):
            dt = factor * self._params.dt
            running_model = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
            running_model.differential.armature = self._robot_models.armature
            running_model_list.extend([running_model] * dts)
        return running_model_list

    def create_terminal_model(self) -> crocoddyl.ActionModelAbstract:
        terminal_cost_model = crocoddyl.CostModelSum(self._state)
        x_residual, _, frame_placement_residual = self.create_residuals()
        x_reg_weights = np.ones(
            self._robot_models.robot_model.nq + self._robot_models.robot_model.nv
        )
        x_reg_cost = crocoddyl.CostModelResidual(
            self._state,
            crocoddyl.ActivationModelWeightedQuad(x_reg_weights),
            x_residual,
        )
        ee_pose_cost = crocoddyl.CostModelResidual(
            self._state,
            crocoddyl.ActivationModelWeightedQuad(np.ones(6)),
            frame_placement_residual,
        )

        for col_idx in range(len(self._collision_model.collisionPairs)):
            activation = ActivationModelQuadExp(
                1, self._params.activation_distance_threshold**2
            )
            distance_coll_residual = ResidualDistanceCollision(
                self._state, 7, self._collision_model, col_idx
            )
            distance_coll_cost = crocoddyl.CostModelResidual(
                self._state, activation, distance_coll_residual
            )
            terminal_cost_model.addCost(f"distColl{col_idx}", distance_coll_cost, 1.0)
        terminal_cost_model.addCost("stateReg", x_reg_cost, 1.0)
        terminal_cost_model.addCost("goalTracking", ee_pose_cost, 1.0)
        constraints = self._get_collision_constraints(active=True)
        terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self._state, self._actuation, terminal_cost_model, constraints
        )

        terminal_model = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 1.0)
        terminal_model.differential.armature = self._robot_models.armature
        return terminal_model

    def _get_collision_constraints(self, active=True):
        """Returns the collision constraints model manager."""
        constraint_model_manager = crocoddyl.ConstraintModelManager(
            self._state, self.nq
        )
        for col_idx in range(len(self._collision_model.collisionPairs)):
            distance_coll_residual = ResidualDistanceCollision(
                self._state, 7, self._collision_model, col_idx
            )
            collision_constraint = crocoddyl.ConstraintModelResidual(
                self._state,
                distance_coll_residual,
                np.array([self._params.collision_safety_margin]),
                np.array([np.inf]),
            )
            # Adding the constraint to the constraint manager
            constraint_model_manager.addConstraint(
                "col_term_" + str(col_idx), collision_constraint, active=active
            )
        return constraint_model_manager

    def get_distance_collision_residuals(self) -> Union[npt.NDArray[np.float64], None]:
        """Return distance collision residuals if Crocoddyl's problem use it."""
        nb_collision_pairs = len(self._collision_model.collisionPairs)
        if nb_collision_pairs != 0:
            coll_residuals = np.zeros((self._params.horizon_size, nb_collision_pairs))
            for node_idx in range(self._params.horizon_size - 1):
                constraints_residual_dict = self._solver.problem.runningDatas[
                    node_idx
                ].differential.constraints.constraints.todict()
                for coll_pair_idx, constraint_key in enumerate(
                    constraints_residual_dict.keys()
                ):
                    coll_residuals[node_idx, coll_pair_idx] = constraints_residual_dict[
                        constraint_key
                    ].residual.r[0]
            constraints_residual_dict = self._solver.problem.terminalData.differential.constraints.constraints.todict()
            for coll_pair_idx, constraint_key in enumerate(
                constraints_residual_dict.keys()
            ):
                coll_residuals[self._params.horizon_size - 1, coll_pair_idx] = (
                    constraints_residual_dict[constraint_key].residual.r[0]
                )
            return coll_residuals
        else:
            return None

    def set_reference_weighted_trajectory(
        self, reference_weighted_trajectory: list[WeightedTrajectoryPoint]
    ):
        """Set the reference trajectory for the OCP."""
        super().set_reference_weighted_trajectory(reference_weighted_trajectory)
        # Modify running costs reference and weights
        for i in range(self.horizon_size - 1):
            model = self._solver.problem.runningModels[i]

            state_ref = np.concatenate(
                (
                    reference_weighted_trajectory[i].point.robot_configuration,
                    reference_weighted_trajectory[i].point.robot_velocity,
                )
            )
            state_weights = np.concatenate(
                (
                    reference_weighted_trajectory[i].weights.w_robot_configuration,
                    reference_weighted_trajectory[i].weights.w_robot_velocity,
                )
            )
            self.modify_cost_reference_and_weights(
                model, "stateReg", state_ref, state_weights
            )

            u_ref = reference_weighted_trajectory[i].point.robot_effort
            u_weights = reference_weighted_trajectory[i].weights.w_robot_effort
            self.modify_cost_reference_and_weights(model, "ctrlReg", u_ref, u_weights)

            ee_names = list(
                reference_weighted_trajectory[i].weights.w_end_effector_poses.keys()
            )
            if len(ee_names) > 1:
                raise ValueError("Only one end-effector tracking reference is allowed.")
            ee_name = ee_names[0]
            ee_ref = reference_weighted_trajectory[i].point.end_effector_poses[ee_name]
            ee_weights = reference_weighted_trajectory[-1].weights.w_end_effector_poses[
                ee_name
            ]
            self.modify_cost_reference_and_weights(
                model, "goalTracking", ee_ref, ee_weights
            )

            ee_id = self._robot_models.robot_model.getFrameId(ee_name)
            model.differential.costs.costs["goalTracking"].cost.residual.id = ee_id

        # Modify terminal costs reference and weights
        model = self._solver.problem.terminalModel

        state_ref = np.concatenate(
            (
                reference_weighted_trajectory[-1].point.robot_configuration,
                reference_weighted_trajectory[-1].point.robot_velocity,
            )
        )
        state_weights = np.concatenate(
            (
                reference_weighted_trajectory[-1].weights.w_robot_configuration,
                reference_weighted_trajectory[-1].weights.w_robot_velocity,
            )
        )
        self.modify_cost_reference_and_weights(
            model, "stateReg", state_ref, state_weights
        )
        # Modify end effector frame cost
        ee_names = list(
            reference_weighted_trajectory[-1].weights.w_end_effector_poses.keys()
        )
        if len(ee_names) > 1:
            raise ValueError("Only one end-effector tracking reference is allowed.")
        ee_name = ee_names[0]
        ee_ref = reference_weighted_trajectory[-1].point.end_effector_poses[ee_name]
        ee_weights = reference_weighted_trajectory[-1].weights.w_end_effector_poses[
            ee_name
        ]
        self.modify_cost_reference_and_weights(
            model, "goalTracking", ee_ref, ee_weights
        )

        ee_id = self._robot_models.robot_model.getFrameId(ee_name)
        model.differential.costs.costs["goalTracking"].cost.residual.id = ee_id

    def solve(
        self,
        x0: npt.NDArray[np.float64],
        x_warmstart: list[npt.NDArray[np.float64]],
        u_warmstart: list[npt.NDArray[np.float64]],
    ) -> None:
        super().solve(x0, x_warmstart, u_warmstart)
        self._debug_data.collision_distance_residuals = (
            self.get_distance_collision_residuals()
        )
