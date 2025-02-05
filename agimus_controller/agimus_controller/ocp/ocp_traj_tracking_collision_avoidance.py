import crocoddyl
import numpy as np
import pinocchio as pin

from agimus_controller.ocp_base_croco import OCPBaseCroco
from agimus_controller.trajectory import WeightedTrajectoryPoint

from colmpc import ActivationModelQuadExp, ResidualDistanceCollision


class OCPCrocoTrajTrackCollAvoidance(OCPBaseCroco):
    def create_residuals(self):
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
        for factor_idx, dts in enumerate(self._params.dt_factor_n_seq.dts):
            factor = self._params.dt_factor_n_seq.factors[factor_idx]
            dt = factor * self._params.dt
            for _ in range(dts):
                running_cost_model = crocoddyl.CostModelSum(self._state)
                x_residual, u_residual, frame_placement_residual = (
                    self.create_residuals()
                )
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
                    running_cost_model.addCost(
                        f"distColl{col_idx}", distance_coll_cost, 1.0
                    )
                running_cost_model.addCost("stateReg", x_reg_cost, 1.0)
                running_cost_model.addCost("ctrlReg", u_reg_cost, 1.0)
                running_cost_model.addCost("goalTracking", ee_pose_cost, 1.0)

                constraints = self.get_collision_constraints(active=True)
                running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
                    self._state, self._actuation, running_cost_model, constraints
                )
                running_model = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
                running_model.differential.armature = self._robot_models.armature
                running_model_list.append(running_model)
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
        constraints = self.get_collision_constraints(active=True)
        terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self._state, self._actuation, terminal_cost_model, constraints
        )

        terminal_model = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.0)
        terminal_model.differential.armature = self._robot_models.armature
        return terminal_model

    def get_collision_constraints(self, active=True):
        """Returns the collision constraints model manager."""
        constraint_model_manager = crocoddyl.ConstraintModelManager(
            self._state, self.nq
        )
        if len(self._collision_model.collisionPairs) != 0:
            for col_idx in range(len(self._collision_model.collisionPairs)):
                collision_constraint = self._get_collision_constraint_residual(col_idx)
                # Adding the constraint to the constraint manager
                constraint_model_manager.addConstraint(
                    "col_term_" + str(col_idx), collision_constraint, active=active
                )
        return constraint_model_manager

    def _get_collision_constraint_residual(
        self, col_idx: int
    ) -> "crocoddyl.ConstraintModelResidual":
        """Returns the collision constraint model residual."""
        distance_coll_residual = ResidualDistanceCollision(
            self._state, 7, self._collision_model, col_idx
        )
        return crocoddyl.ConstraintModelResidual(
            self._state,
            distance_coll_residual,
            np.array([self._params.collision_safety_margin]),
            np.array([np.inf]),
        )

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
                iter(reference_weighted_trajectory[i].weights.w_end_effector_poses)
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
            iter(reference_weighted_trajectory[-1].weights.w_end_effector_poses)
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
