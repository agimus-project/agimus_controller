import crocoddyl
import pinocchio as pin
import numpy as np
import example_robot_data
import time
import mim_solvers


class Problem:
    def __init__(self, x_plan, a_plan, robot_name):
        self.x_cost = 1e-1
        self.u_cost = 1e-4
        self.grip_cost = 1e6
        self.xlim_cost = 0
        self.vel_cost = 0
        self.use_mim = False
        self.null_speed_path_idxs = []

        #    1,
        #    3,
        # ]  # sub path indexes where we desire null speed at the last node
        self.robot = example_robot_data.load(robot_name)
        if robot_name in ["ur3", "ur5", "ur10"]:
            self.last_joint_name = "wrist_3_joint"
        elif robot_name == "panda":
            self.last_joint_name = "panda_joint7"
            locked_joints = [
                self.robot.model.getJointId("panda_finger_joint1"),
                self.robot.model.getJointId("panda_finger_joint2"),
            ]
            robot_model_reduced = pin.buildReducedModel(
                self.robot.model, locked_joints, self.robot.q0
            )
            self.robot.model = robot_model_reduced
        else:
            raise Exception("Unkown robot")
        self.robot_data = self.robot.model.createData()
        self.state = crocoddyl.StateMultibody(self.robot.model)
        self.actuation = crocoddyl.ActuationModelFull(self.state)
        self.DT = 1e-2
        self.nq = self.robot.nq
        self.nv = self.robot.nv

        self.x_plan = x_plan  # hpp's x_plan for the whole trajectory
        self.a_plan = a_plan
        self.u_ref = self.get_uref(x_plan, a_plan)
        self.whole_traj_T = x_plan.shape[0]
        self.T = x_plan.shape[0]
        self.running_models = None
        self.terminal_model = None
        self.solver = None

    def get_uref(self, x_plans, a_plans):
        """Return the reference of control u_ref that compensates gravity."""
        u_ref = np.zeros([x_plans.shape[0] - 1, self.nv])
        """
        for x in self.hpp_paths[path_idx].x_plan:
            pin.computeGeneralizedGravity(
                self.robot.model,
                self.robot_data,
                x[: self.nq],
            )
            u_ref.append(self.robot_data.g.copy())"""
        for idx in range(x_plans.shape[0] - 1):
            x = x_plans[idx, :]
            a = a_plans[idx, :]
            tau = pin.rnea(
                self.robot.model, self.robot.data, x[: self.nq], x[self.nq :], a
            ).copy()
            u_ref[idx, :] = tau[: self.nq]
        return u_ref

    def set_costs(self, grip_cost, x_cost, u_cost, vel_cost=0, xlim_cost=0):
        """Set costs of the ddp problem."""
        self.x_cost = x_cost
        self.u_cost = u_cost
        self.grip_cost = grip_cost
        self.vel_cost = vel_cost
        self.xlim_cost = xlim_cost

    def set_models(self):
        """Set running models and terminal model of the ddp problem."""
        goal_placement_residual = self.get_goal_placement_residual()
        self.set_sub_path_running_models()
        self.set_last_model(goal_placement_residual, True)

    def set_sub_path_running_models(self):
        """Set running models for one sub path of the ddp problem."""

        running_models = []
        x_reg_weights = crocoddyl.ActivationModelWeightedQuad(
            np.array([1] * self.nq + [10] * self.nv) ** 2
        )
        for idx in range(self.T - 1):
            running_cost_model = crocoddyl.CostModelSum(self.state)
            x_ref = self.x_plan[idx, :]
            x_residual = self.get_state_residual(x_ref)
            u_residual = self.get_control_residual(self.u_ref[idx, :])
            xLimit_residual = self.get_xlimit_residual()
            frame_velocity_residual = self.get_velocity_residual(self.last_joint_name)
            placemment_residual = self.get_placement_residual(x_ref[: self.nq])

            running_cost_model.addCost("xReg", x_residual, self.x_cost)
            running_cost_model.addCost("uReg", u_residual, self.u_cost)
            # running_cost_model.addCost("xlimitReg", xLimit_residual, self.xlim_cost)
            running_cost_model.addCost("vel", frame_velocity_residual, self.vel_cost)
            running_cost_model.addCost(
                "gripperPose", placemment_residual, 0
            )  # useful for mpc to reset ocp
            running_models.append(
                crocoddyl.IntegratedActionModelEuler(
                    crocoddyl.DifferentialActionModelFreeFwdDynamics(
                        self.state, self.actuation, running_cost_model
                    ),
                    self.DT,
                )
            )
        self.running_models = running_models

    def set_last_model(self, goal_placement_residual, is_terminal_model=True):
        """Set last model for a sub path."""
        u_ref = self.u_ref[-1, :]
        if self.use_mim:
            last_model = self.get_last_model_with_mim()
        else:

            last_model = self.get_last_model_without_mim(
                goal_placement_residual, self.x_plan[-1, :], u_ref
            )
        if is_terminal_model:
            self.terminal_model = last_model
        else:
            self.running_models.append(last_model)

    def get_last_model_without_mim(self, goal_placement_residual, x_ref, u_ref):
        """Return last model without constraints."""
        running_cost_model = crocoddyl.CostModelSum(self.state)
        running_cost_model.addCost(
            "gripperPose", goal_placement_residual, self.grip_cost
        )
        vel_cost = self.get_velocity_residual(self.last_joint_name)
        if np.linalg.norm(x_ref[self.nq :]) < 1e-6:
            running_cost_model.addCost("vel", vel_cost, self.grip_cost)
        else:
            running_cost_model.addCost("vel", vel_cost, 0)
        x_residual = self.get_state_residual(x_ref)
        running_cost_model.addCost("xReg", x_residual, 0)

        if u_ref is not None:
            u_reg_cost = self.get_control_residual(u_ref)
            running_cost_model.addCost("uReg", u_reg_cost, self.u_cost)
        else:
            u_reg_cost = self.get_control_residual(np.zeros([self.nq, 1]))
            running_cost_model.addCost("uReg", u_reg_cost, 0)
        return crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(
                self.state, self.actuation, running_cost_model
            ),
            self.DT,
        )

    def get_last_model_with_mim(self):
        """Return last model for a sub path with constraints for mim_solvers."""
        running_cost_model = crocoddyl.CostModelSum(self.state)
        constraints = crocoddyl.ConstraintModelManager(self.state, self.nv)

        # add placement constraint
        placement_residual = self.get_placement_residual(self.x_plan[-1, : self.nq])
        placement_constraint = crocoddyl.ConstraintModelResidual(
            self.state,
            placement_residual,
            np.array([0] * 12),
            np.array([1e-3] * 12),
        )
        constraints.addConstraint("gripperPose", placement_constraint)

        # add torque constraint
        """
        torque_residual = crocoddyl.ResidualModelJointEffort(
            self.state,
            self.actuation,
            np.array([0] * 6),  # np.array([0] * 6)
        )
        torque_constraint = crocoddyl.ConstraintModelResidual(
            self.state,
            torque_residual,
            np.array([-87] * 4 + [-12] * 3),
            np.array([87] * 4 + [12] * 3),
        )
        constraints.addConstraint("JointsEfforts", torque_constraint)

        # add velocities constraints

        vref = pin.Motion.Zero()

        joint_vel_residual = crocoddyl.ResidualModelFrameVelocity(
            self.state,
            self.robot.model.getFrameId(self.last_joint_name),
            vref,
            pin.LOCAL,
        )
        joint_vel_constraint = crocoddyl.ConstraintModelResidual(
            self.state,
            joint_vel_residual,
            np.array([0] * self.nv),
            np.array([1e-3] * self.nv),
        )
        constraints.addConstraint("shoulder_lift_joint_velocity", joint_vel_constraint)
        
        for joint_name in self.robot.model.names:
            joint_vel_residual = crocoddyl.ResidualModelFrameVelocity(
                self.state,
                self.robot.model.getFrameId(joint_name),
                vref,
                pin.LOCAL,
            )
            joint_vel_constraint = crocoddyl.ConstraintModelResidual(
                self.state,
                joint_vel_residual,
                np.array([-1e-6] * self.nv),
                np.array([1e-6] * self.nv),
            )
            constraints.addConstraint(f"{joint_name}_velocity", joint_vel_constraint)
        """
        return crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(
                self.state, self.actuation, running_cost_model, constraints
            ),
            self.DT,
        )

    def get_goal_placement_residual(self):
        """Return vector of tracking costs for each sub path."""

        return self.get_placement_residual(self.x_plan[-1, : self.nq])

    def get_placement_residual(self, q):
        """Return placement residual to the last position of the sub path."""
        target = self.robot.placement(q, self.nq).copy()
        return crocoddyl.CostModelResidual(
            self.state,
            crocoddyl.ResidualModelFramePlacement(
                self.state, self.robot.model.getFrameId(self.last_joint_name), target
            ),
        )

    def get_velocity_residual(self, joint_name):
        """Return velocity residual of desired joint."""
        vref = pin.Motion.Zero()
        return crocoddyl.CostModelResidual(
            self.state,
            crocoddyl.ResidualModelFrameVelocity(
                self.state,
                self.robot.model.getFrameId(joint_name),
                vref,
                pin.WORLD,
            ),
        )

    def get_control_residual(self, uref):
        return crocoddyl.CostModelResidual(
            self.state, crocoddyl.ResidualModelControl(self.state, uref)
        )

    def get_state_residual(self, xref):
        return crocoddyl.CostModelResidual(
            self.state,  # x_reg_weights,
            crocoddyl.ResidualModelState(self.state, xref, self.actuation.nu),
        )

    def get_xlimit_residual(self):
        """Return velocity residual of desired joint."""
        return crocoddyl.CostModelResidual(
            self.state,
            crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(self.state.lb, self.state.ub)
            ),
            crocoddyl.ResidualModelState(
                self.state,
                np.array([0] * (self.nq + self.nv)),
                self.actuation.nu,
            ),
        )

    def get_translation_residual(self, path_idx):
        """Return translation residual to the last position of the sub path."""
        q_final = self.x_plan[-1, : self.nq]
        target = self.robot.placement(q_final, self.nq)
        return crocoddyl.ResidualModelFrameTranslation(
            self.state,
            self.robot.model.getFrameId(self.last_joint_name),
            target.translation,
        )

    def create_whole_problem(self):
        x0 = self.x_plan[0, :]

        self.whole_problem = crocoddyl.ShootingProblem(
            x0, self.running_models, self.terminal_model
        )
        self.whole_problem_models = list(self.whole_problem.runningModels)

    def create_problem(self, T):
        x0 = (
            self.whole_problem.runningModels[0]
            .differential.costs.costs["xReg"]
            .cost.residual.reference
        )
        models = []
        for i in range(T - 1):
            models.append(self.whole_problem.runningModels[i].copy())

        terminal_model = self.whole_problem.runningModels[T - 1].copy()
        return crocoddyl.ShootingProblem(x0, models, terminal_model)

    def update_cost(self, model, new_model, cost_name):
        model.differential.costs.costs[cost_name].cost.residual.reference = (
            new_model.differential.costs.costs[cost_name].cost.residual.reference.copy()
        )
        new_weight = new_model.differential.costs.costs[cost_name].weight
        model.differential.costs.costs[cost_name].weight = new_weight
        if new_weight == 0:
            model.differential.costs.changeCostStatus(cost_name, False)
        else:
            model.differential.costs.changeCostStatus(cost_name, True)

    def update_model(self, model, new_model):
        self.update_cost(model, new_model, "xReg")
        self.update_cost(model, new_model, "gripperPose")
        self.update_cost(model, new_model, "vel")
        self.update_cost(model, new_model, "uReg")

    def update_terminal_model(self, model, new_model):
        model.differential.costs.costs["xReg"].weight = 0
        model.differential.costs.costs["gripperPose"].cost.residual.reference = (
            new_model.differential.costs.costs["gripperPose"].cost.residual.reference
        )  # FIXME uniquement utile si le coût est != de 0 donc à changer je suppose
        model.differential.costs.costs["gripperPose"].weight = self.grip_cost
        model.differential.costs.costs["vel"].weight = (
            new_model.differential.costs.costs["vel"].weight
        )
        if "ureg" in new_model.differential.costs.costs.todict().keys():
            model.differential.costs.costs["uReg"].cost.residual.reference = (
                new_model.differential.costs.costs["uReg"].cost.residual.reference
            )
        elif "ureg" in model.differential.costs.costs.todict().keys():

            model.differential.costs.costs["uReg"].weight = 0

    def reset_ocp(self, x, next_node_idx):
        self.solver.problem.x0 = x
        runningModels = list(self.solver.problem.runningModels)
        start_idx = next_node_idx - (len(self.solver.problem.runningModels) + 1)
        for node_idx in range(len(runningModels) - 1):
            self.update_model(
                runningModels[node_idx],
                runningModels[node_idx + 1],
            )
        if next_node_idx >= self.whole_traj_T - 1:
            self.update_model(
                runningModels[-1],
                self.whole_problem.runningModels[-1].copy(),
            )
        else:
            self.update_model(
                runningModels[-1],
                self.whole_problem.runningModels[next_node_idx - 1].copy(),
            )
        if next_node_idx < self.whole_traj_T - 1:
            self.update_model(
                self.solver.problem.terminalModel,
                self.whole_problem.runningModels[next_node_idx].copy(),
            )

        else:
            self.update_model(
                self.solver.problem.terminalModel,
                self.whole_problem.terminalModel.copy(),
            )

    def run_solver(self, problem, xs_init, us_init, max_iter, set_callback=False):
        """
        Run ddp solver
        start_idx : hpp's sub path idx from which we start to run ddp on
        terminal_idx : hpp's sub path idx from which we end to run ddp.
        """
        # Creating the solver for this OC problem, defining a logger
        if self.use_mim:
            solver = mim_solvers.SolverCSQP(problem)
            solver.use_filter_line_search = True
            solver.termination_tolerance = 1e-3
            solver.max_qp_iters = 100
            # solver.eps_rel = 0
            # solver.eps_abs = 1e-6
            # solver.with_callbacks = True
            # solver.reset_rho = True
            # solver.reset_y = True
        else:
            solver = crocoddyl.SolverFDDP(problem)
            solver.use_filter_line_search = True
            solver.termination_tolerance = 1e-3
            if set_callback:
                solver.setCallbacks([crocoddyl.CallbackVerbose()])

        # Warm start with hpp trajectory then solve
        # solver.setCandidate(self.x_plan, self.u_ref, False)
        start = time.time()
        solver.solve(xs_init, us_init, max_iter)
        end = time.time()
        # print("solve duration ", end - start)
        self.solver = solver
