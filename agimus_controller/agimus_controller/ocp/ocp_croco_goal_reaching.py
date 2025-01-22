import crocoddyl
import numpy as np
import pinocchio as pin

from agimus_controller.ocp_base_croco import OCPBaseCroco
from agimus_controller.trajectory import WeightedTrajectoryPoint


class OCPCrocoGoalReaching(OCPBaseCroco):
    def create_running_model_list(self) -> list[crocoddyl.ActionModelAbstract]:
        running_model_list = []
        for _ in range(self._ocp_params.horizon_size - 1):
            # Running cost model
            running_cost_model = crocoddyl.CostModelSum(self._state)

            ### Creation of cost terms
            # State Regularization cost
            x_reg_weights = np.ones(
                self._robot_models.robot_model.nq + self._robot_models.robot_model.nv
            )
            x_residual = crocoddyl.ResidualModelState(
                self._state,
                np.concatenate(
                    (
                        pin.neutral(self._robot_models.robot_model),
                        np.zeros(self._robot_models.robot_model.nv),
                    )
                ),
            )
            x_reg_cost = crocoddyl.CostModelResidual(
                self._state,
                crocoddyl.ActivationModelWeightedQuad(x_reg_weights),
                x_residual,
            )
            # Control Regularization cost
            u_reg_weights = np.ones(self._robot_models.robot_model.nv)
            u_residual = crocoddyl.ResidualModelControl(self._state)
            u_reg_cost = crocoddyl.CostModelResidual(
                self._state,
                crocoddyl.ActivationModelWeightedQuad(u_reg_weights),
                u_residual,
            )

            # End effector frame cost
            frame_cost_weights = np.ones(6)
            framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
                self._state,
                self._robot_models.robot_model.getFrameId("panda_hand_tcp"),
                pin.SE3.Identity(),
            )
            goalTrackingCost = crocoddyl.CostModelResidual(
                self._state,
                crocoddyl.ActivationModelWeightedQuad(frame_cost_weights),
                framePlacementResidual,
            )

            running_cost_model.addCost("stateReg", x_reg_cost, 1.0)
            running_cost_model.addCost("ctrlReg", u_reg_cost, 1.0)
            running_cost_model.addCost("goalTracking", goalTrackingCost, 1.0)
            # Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
            running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
                self._state,
                self._actuation,
                running_cost_model,
            )
            running_model = crocoddyl.IntegratedActionModelEuler(
                running_DAM,
            )
            running_model.differential.armature = self._robot_models.armature

            running_model_list.append(running_model)
        return running_model_list

    def create_terminal_model(self) -> crocoddyl.ActionModelAbstract:
        # Terminal cost models
        terminal_cost_model = crocoddyl.CostModelSum(self._state)

        ### Creation of cost terms
        # State Regularization cost
        x_reg_weights = np.ones(
            self._robot_models.robot_model.nq + self._robot_models.robot_model.nv
        )
        x_residual = crocoddyl.ResidualModelState(
            self._state,
            np.concatenate(
                (
                    pin.neutral(self._robot_models.robot_model),
                    np.zeros(self._robot_models.robot_model.nv),
                )
            ),
        )
        x_reg_cost = crocoddyl.CostModelResidual(
            self._state,
            crocoddyl.ActivationModelWeightedQuad(x_reg_weights),
            x_residual,
        )

        # End effector frame cost
        frame_cost_weights = np.ones(6)
        framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
            self._state,
            self._robot_models.robot_model.getFrameId("panda_hand_tcp"),
            pin.SE3.Identity(),
        )

        goalTrackingCost = crocoddyl.CostModelResidual(
            self._state,
            crocoddyl.ActivationModelWeightedQuad(frame_cost_weights),
            framePlacementResidual,
        )

        terminal_cost_model.addCost("stateReg", x_reg_cost, 1.0)
        terminal_cost_model.addCost("goalTracking", goalTrackingCost, 1.0)
        # Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
        terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self._state,
            self._actuation,
            terminal_cost_model,
        )

        terminal_model = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.0)
        terminal_model.differential.armature = self._robot_models.armature
        return terminal_model

    def set_reference_weighted_trajectory(
        self, reference_weighted_trajectory: list[WeightedTrajectoryPoint]
    ):
        """Set the reference trajectory for the OCP."""

        # Modify running costs reference and weights
        for i in range(self.horizon_size - 1):
            # Modifying the state regularization cost
            state_reg = self._solver.problem.runningModels[i].differential.costs.costs[
                "stateReg"
            ]
            state_reg.cost.residual.reference = np.concatenate(
                (
                    reference_weighted_trajectory[i].point.robot_configuration,
                    reference_weighted_trajectory[i].point.robot_velocity,
                )
            )
            # Modify running cost weight
            state_reg.cost.activation.weights = np.concatenate(
                (
                    reference_weighted_trajectory[i].weight.w_robot_configuration,
                    reference_weighted_trajectory[i].weight.w_robot_velocity,
                )
            )
            # state_reg.weight = reference_weighted_trajectory[
            # i
            # ].weight.w_robot_configuration
            # Modify control regularization cost
            u_ref = reference_weighted_trajectory[i].point.robot_effort
            ctrl_reg = self._solver.problem.runningModels[i].differential.costs.costs[
                "ctrlReg"
            ]
            ctrl_reg.cost.residual.reference = u_ref
            # Modify running cost weight
            # ctrl_reg.weight = reference_weighted_trajectory[i].weight.w_robot_effort
            ctrl_reg.cost.activation.weights = reference_weighted_trajectory[
                i
            ].weight.w_robot_effort
            # Modify end effector frame cost
            ee_cost = self._solver.problem.runningModels[i].differential.costs.costs[
                "goalTracking"
            ]
            # ee_cost.weight = reference_weighted_trajectory[i].weight.w_end_effector_poses[
            #     "panda_hand_tcp"
            # ]
            ee_cost.cost.activation.weights = reference_weighted_trajectory[
                i
            ].weight.w_end_effector_poses["panda_hand_tcp"]
            ee_cost.cost.residual.reference = reference_weighted_trajectory[
                i
            ].point.end_effector_poses["panda_hand_tcp"]

        # Modify terminal costs reference and weights
        state_reg = self._solver.problem.terminalModel.differential.costs.costs[
            "stateReg"
        ]
        state_reg.cost.residual.reference = np.concatenate(
            (
                reference_weighted_trajectory[-1].point.robot_configuration,
                reference_weighted_trajectory[-1].point.robot_velocity,
            )
        )

        # state_reg.weight = reference_weighted_trajectory[-1].weight.w_robot_configuration
        state_reg.cost.activation.weights = np.concatenate(
            (
                reference_weighted_trajectory[i].weight.w_robot_configuration,
                reference_weighted_trajectory[i].weight.w_robot_velocity,
            )
        )
        # Modify end effector frame cost
        ee_cost = self._solver.problem.runningModels[-1].differential.costs.costs[
            "goalTracking"
        ]
        # ee_cost.weight = reference_weighted_trajectory[-1].weight.w_end_effector_poses[
        #     "panda_hand_tcp"
        # ]
        ee_cost.cost.residual.reference = reference_weighted_trajectory[
            -1
        ].point.end_effector_poses["panda_hand_tcp"]
        ee_cost.cost.activation.weights = reference_weighted_trajectory[
            i
        ].weight.w_end_effector_poses["panda_hand_tcp"]
