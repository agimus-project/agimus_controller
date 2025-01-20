import crocoddyl
import numpy as np
import pinocchio as pin

from agimus_controller.ocp_base_croco import OCPBaseCroco


class OCPCrocoJointState(OCPBaseCroco):
    def create_running_model_list(self) -> list[crocoddyl.ActionModelAbstract]:
        running_model_list = []
        for _ in range(self._ocp_params.horizon_size - 1):
            # Running cost model
            running_cost_model = crocoddyl.CostModelSum(self._state)

            ### Creation of cost terms
            # State Regularization cost
            x_residual = crocoddyl.ResidualModelState(
                self._state,
                np.concatenate(
                    (
                        pin.neutral(self._robot_models.robot_model),
                        np.zeros(self._robot_models.robot_model.nv),
                    )
                ),
            )
            x_reg_cost = crocoddyl.CostModelResidual(self._state, x_residual)
            # Control Regularization cost
            u_residual = crocoddyl.ResidualModelControl(self._state)
            u_reg_cost = crocoddyl.CostModelResidual(self._state, u_residual)

            # End effector frame cost
            framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
                self._state,
                self._robot_models.robot_model.getFrameId("panda_hand_tcp"),
                pin.SE3.Identity(),
            )

            goalTrackingCost = crocoddyl.CostModelResidual(
                self._state, framePlacementResidual
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
        x_residual = crocoddyl.ResidualModelState(
            self._state,
            np.concatenate(
                (
                    pin.neutral(self._robot_models.robot_model),
                    np.zeros(self._robot_models.robot_model.nv),
                )
            ),
        )
        x_reg_cost = crocoddyl.CostModelResidual(self._state, x_residual)

        # End effector frame cost
        framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
            self._state,
            self._robot_models.robot_model.getFrameId("panda_hand_tcp"),
            pin.SE3.Identity(),
        )

        goalTrackingCost = crocoddyl.CostModelResidual(
            self._state, framePlacementResidual
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

    def set_reference_weighted_trajectory(self, weighted_trajectory_points):
        """Set the reference trajectory for the OCP."""

        # Modify running costs reference and weights
        for t in range(self.horizon_size - 1):
            # Modifying the state regularization cost
            xref = np.concatenate(
                (
                    weighted_trajectory_points[t].point.robot_configuration,
                    weighted_trajectory_points[t].point.robot_velocity,
                )
            )
            state_reg = self._solver.problem.runningModels[t].differential.costs.costs[
                "stateReg"
            ]
            state_reg.cost.residual.reference = xref
            # Modify running cost weight
            state_reg.weight = weighted_trajectory_points[
                t
            ].weight.w_robot_configuration
            # Modify control regularization cost
            u_ref = weighted_trajectory_points[t].point.robot_effort
            ctrl_reg = self._solver.problem.runningModels[t].differential.costs.costs[
                "ctrlReg"
            ]
            ctrl_reg.cost.residual.reference = u_ref
            # Modify running cost weight
            ctrl_reg.weight = weighted_trajectory_points[t].weight.w_robot_effort

            # Modify end effector frame cost
            ee_cost = self._solver.problem.runningModels[t].differential.costs.costs[
                "goalTracking"
            ]
            ee_cost.weight = weighted_trajectory_points[t].weight.w_end_effector_poses[
                "panda_hand_tcp"
            ]
            ee_cost.cost.residual.reference = weighted_trajectory_points[
                t
            ].point.end_effector_poses["panda_hand_tcp"]

        # Modify terminal costs reference and weights
        xref = np.concatenate(
            (
                weighted_trajectory_points[-1].point.robot_configuration,
                weighted_trajectory_points[-1].point.robot_velocity,
            )
        )

        state_reg = self._solver.problem.terminalModel.differential.costs.costs[
            "stateReg"
        ]
        state_reg.cost.residual.reference = xref
        state_reg.weight = weighted_trajectory_points[-1].weight.w_robot_configuration

        # Modify end effector frame cost
        ee_cost = self._solver.problem.runningModels[-1].differential.costs.costs[
            "goalTracking"
        ]
        ee_cost.weight = weighted_trajectory_points[-1].weight.w_end_effector_poses[
            "panda_hand_tcp"
        ]
        ee_cost.cost.residual.reference = weighted_trajectory_points[
            -1
        ].point.end_effector_poses["panda_hand_tcp"]
