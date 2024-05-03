import crocoddyl
import pinocchio as pin
import numpy as np
import example_robot_data
import time
import mim_solvers


class SubPath:
    def __init__(self, path):
        if path is not None:
            self.path = path  # hpp path
            self.T = int(20 * self.path.length())
            self.x_plan = self.get_xplan()
        self.u_ref = None
        self.running_models = None
        self.terminal_model = None

    def get_xplan(self):
        """Return x_plan, the state trajectory of hpp."""
        x_plan = []
        total_time = self.path.length()
        if self.T == 0:
            return x_plan
        for iter in range(self.T + 1):
            iter_time = total_time * iter / self.T
            q_t = np.array(self.path.call(iter_time)[0][:6])
            v_t = np.array(self.path.derivative(iter_time, 1)[:6])
            x_plan.append(np.concatenate([q_t, v_t]))
        return x_plan


class Problem:
    def __init__(self, ps, robot_name):
        self.x_cost = 1e-1
        self.u_cost = 1e-4
        self.grip_cost = 1e6
        self.xlim_cost = 0.01
        self.vel_cost = 0.01
        self.null_speed_path_idxs = [
            1,
            3,
        ]  # sub path indexes where we desire null speed at the last node
        self.robot = example_robot_data.load(robot_name)
        self.robot_data = self.robot.model.createData()
        self.state = crocoddyl.StateMultibody(self.robot.model)
        self.actuation_model = crocoddyl.ActuationModelFull(self.state)
        self.DT = 5e-2
        self.nq = self.robot.nq
        self.nv = self.robot.nv

        self.hpp_paths = []
        if ps is not None:
            self.p = ps.client.basic.problem.getPath(ps.numberPaths() - 1)
            for i in range(self.p.numberPaths()):
                new_path = SubPath(self.p.pathAtRank(i))
                if new_path.T > 0:
                    self.hpp_paths.append(new_path)
            self.nb_paths = len(self.hpp_paths)  # number of sub paths
            self.solver = None

    def get_uref(self, path_idx):
        """Return the reference of control u_ref that compensates gravity."""
        u_ref = []
        for x in self.hpp_paths[path_idx].x_plan:
            pin.computeGeneralizedGravity(
                self.robot.model,
                self.robot_data,
                x[: self.nq],
            )
            u_ref.append(self.robot_data.g.copy())
        return u_ref

    def set_costs(self, grip_cost, x_cost, u_cost, vel_cost=0, xlim_cost=0):
        """Set costs of the ddp problem."""
        self.x_cost = x_cost
        self.u_cost = u_cost
        self.grip_cost = grip_cost
        self.vel_cost = vel_cost
        self.xlim_cost = xlim_cost

    def set_models(self, terminal_paths_idxs, use_mim=False):
        """Set running models and terminal model of the ddp problem."""
        goal_tracking_costs = self.get_tracking_costs()
        for path_idx in range(self.nb_paths):
            self.hpp_paths[path_idx].u_ref = self.get_uref(path_idx)
            self.set_sub_path_running_models(path_idx)
            self.set_last_model(
                terminal_paths_idxs, goal_tracking_costs[path_idx], path_idx, use_mim
            )

    def set_sub_path_running_models(self, path_idx):
        """Set running models for one sub path of the ddp problem."""

        running_models = []
        x_reg_weights = crocoddyl.ActivationModelWeightedQuad(
            np.array([1] * self.nq + [10] * self.nv) ** 2
        )

        for idx in range(self.hpp_paths[path_idx].T):
            running_cost_model = crocoddyl.CostModelSum(self.state)
            x_reg_cost = crocoddyl.CostModelResidual(
                self.state,  # x_reg_weights,
                crocoddyl.ResidualModelState(
                    self.state,
                    self.hpp_paths[path_idx].x_plan[idx],  # np.array([0] * 12)
                ),
            )
            u_reg_cost = crocoddyl.CostModelResidual(
                self.state,
                crocoddyl.ResidualModelControl(
                    self.state, self.hpp_paths[path_idx].u_ref[idx]
                ),
            )
            xLimitCost = crocoddyl.CostModelResidual(
                self.state,
                crocoddyl.ActivationModelQuadraticBarrier(
                    crocoddyl.ActivationBounds(self.state.lb, self.state.ub)
                ),
                crocoddyl.ResidualModelState(
                    self.state,
                    np.array([0] * (self.nq + self.nv)),
                    self.actuation_model.nu,
                ),
            )

            vref = pin.Motion.Zero()
            frameVelocityCost = crocoddyl.CostModelResidual(
                self.state,
                crocoddyl.ResidualModelFrameVelocity(
                    self.state,
                    self.robot.model.getFrameId("wrist_3_joint"),
                    vref,
                    pin.WORLD,
                ),
            )

            running_cost_model.addCost("xReg", x_reg_cost, self.x_cost)
            running_cost_model.addCost("uReg", u_reg_cost, self.u_cost)
            # running_cost_model.addCost("xlimitReg", xLimitCost, self.xlim_cost)
            # running_cost_model.addCost("vel", frameVelocityCost, self.vel_cost)

            # running_cost_model.addCost("xlimitReg", u_reg_cost, 5)
            running_models.append(
                crocoddyl.IntegratedActionModelEuler(
                    crocoddyl.DifferentialActionModelFreeFwdDynamics(
                        self.state, self.actuation_model, running_cost_model
                    ),
                    self.DT,
                )
            )
        self.hpp_paths[path_idx].running_models = running_models

    def set_last_model(
        self, terminal_paths_idxs, goal_tracking_cost, path_idx, use_mim=False
    ):
        """Set last model for a sub path."""
        if use_mim:
            last_model = self.get_last_model_with_mim(path_idx)
        else:
            last_model = self.get_last_model_without_mim(
                terminal_paths_idxs, goal_tracking_cost, path_idx
            )
        if path_idx in terminal_paths_idxs:
            self.hpp_paths[path_idx].u_ref.pop()
            self.hpp_paths[path_idx].terminal_model = last_model
        else:
            self.hpp_paths[path_idx].running_models.append(last_model)

    def get_last_model_without_mim(
        self, terminal_paths_idxs, goal_tracking_cost, path_idx
    ):
        """Return last model without constraints."""
        running_cost_model = crocoddyl.CostModelSum(self.state)
        running_cost_model.addCost("gripperPose", goal_tracking_cost, self.grip_cost)
        if path_idx in self.null_speed_path_idxs or path_idx in terminal_paths_idxs:

            vref = pin.Motion.Zero()
            # for joint_name in self.robot.model.names:
            vel_cost = crocoddyl.CostModelResidual(
                self.state,
                crocoddyl.ResidualModelFrameVelocity(
                    self.state,
                    self.robot.model.getFrameId("wrist_3_joint"),
                    vref,
                    pin.LOCAL,
                ),
            )
            running_cost_model.addCost("vel_wrist_3_joint", vel_cost, self.grip_cost)
            """
            vref = pin.Motion.Zero()
            frameVelocityCost = crocoddyl.CostModelResidual(
                self.state,
                crocoddyl.ResidualModelFrameVelocity(
                    self.state,
                    self.robot.model.getFrameId("wrist_3_joint"),
                    vref,
                    pin.WORLD,
                ),
            )
            
            running_cost_model.addCost("vel", frameVelocityCost, self.grip_cost)"""
        if path_idx != 6:
            u_reg_cost = crocoddyl.CostModelResidual(
                self.state,
                crocoddyl.ResidualModelControl(
                    self.state, self.hpp_paths[path_idx].u_ref[-1]
                ),
            )
            running_cost_model.addCost("uReg", u_reg_cost, self.u_cost * 1000)
        """
        xLimitCost = crocoddyl.CostModelResidual(
            self.state,
            crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(self.state.lb, self.state.ub)
            ),
            crocoddyl.ResidualModelState(
                self.state,
                np.array([0] * (self.nq + self.nv)),
                self.actuation_model.nu,
            ),
        )
        running_cost_model.addCost("xlimitReg", xLimitCost, self.xlim_cost / 10)"""
        return crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(
                self.state, self.actuation_model, running_cost_model
            ),
            self.DT,
        )

    def get_last_model_with_mim(self, path_idx):
        """Return last model for a sub path with constraints for mim_solvers."""
        running_cost_model = crocoddyl.CostModelSum(self.state)
        constraints = crocoddyl.ConstraintModelManager(self.state, self.nv)

        # add placement constraint
        if True:  # path_idx == 0:
            placement_residual = self.get_placement_residual(path_idx)
            trans_constraint = crocoddyl.ConstraintModelResidual(
                self.state,
                placement_residual,
                np.array([-1e-6] * 12),
                np.array([1e-6] * 12),
            )
        else:
            placement_residual = self.get_translation_residual(
                path_idx
            )  # self.get_placement_residual(path_idx)
            trans_constraint = crocoddyl.ConstraintModelResidual(
                self.state,
                placement_residual,
                np.array([-1e-6] * 3),
                np.array([1e-6] * 3),
            )
        constraints.addConstraint("gripperPose", trans_constraint)

        # add torque constraint

        torque_residual = crocoddyl.ResidualModelJointEffort(
            self.state,
            self.actuation_model,
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
        """
        joint_vel_residual = crocoddyl.ResidualModelFrameVelocity(
            self.state,
            self.robot.model.getFrameId("wrist_3_joint"),
            vref,
            pin.LOCAL,
        )
        joint_vel_constraint = crocoddyl.ConstraintModelResidual(
            self.state,
            joint_vel_residual,
            np.array([0] * self.nv),
            np.array([1e-5] * self.nv),
        )
        constraints.addConstraint("shoulder_lift_joint_velocity", joint_vel_constraint)
        """
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

        return crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(
                self.state, self.actuation_model, running_cost_model, constraints
            ),
            self.DT,
        )

    def get_tracking_costs(self):
        """Return vector of tracking costs for each sub path."""
        goal_tracking_costs = []
        for path_idx in range(self.nb_paths):
            goal_tracking_costs.append(
                crocoddyl.CostModelResidual(
                    self.state, self.get_placement_residual(path_idx)
                )
            )
        return goal_tracking_costs

    def get_placement_residual(self, path_idx):
        """Return placement residual to the last position of the sub path."""
        q_final = self.hpp_paths[path_idx].x_plan[-1][: self.nq]
        target = self.robot.placement(q_final, self.nq)
        return crocoddyl.ResidualModelFramePlacement(
            self.state, self.robot.model.getFrameId("wrist_3_joint"), target
        )

    def get_translation_residual(self, path_idx):
        """Return translation residual to the last position of the sub path."""
        q_final = self.hpp_paths[path_idx].x_plan[-1][: self.nq]
        target = self.robot.placement(q_final, self.nq)
        return crocoddyl.ResidualModelFrameTranslation(
            self.state,
            self.robot.model.getFrameId("wrist_3_joint"),
            target.translation,
        )

    def run_solver(self, start_idx, terminal_idx, use_mim=False, set_callback=False):
        """
        Run ddp solver
        start_idx : hpp's sub path idx from which we start to run ddp on
        terminal_idx : hpp's sub path idx from which we end to run ddp.
        """
        # Create the problem
        x0 = self.hpp_paths[0].x_plan[0]
        final_running_model = []
        final_x_plan = []
        final_u_ref = []
        for path_idx in range(start_idx, terminal_idx + 1):
            final_running_model += self.hpp_paths[path_idx].running_models
            final_x_plan += self.hpp_paths[path_idx].x_plan
            final_u_ref += self.hpp_paths[path_idx].u_ref
        problem = crocoddyl.ShootingProblem(
            x0, final_running_model, self.hpp_paths[terminal_idx].terminal_model
        )
        # Creating the solver for this OC problem, defining a logger
        if use_mim:
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
        solver.setCandidate(final_x_plan, final_u_ref, False)
        start = time.time()
        solver.solve()
        end = time.time()
        # print("solve duration ", end - start)
        self.solver = solver
