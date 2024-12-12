import crocoddyl
import pinocchio as pin
import numpy as np
import mim_solvers
from colmpc import ResidualDistanceCollision

from agimus_controller.utils.pin_utils import get_ee_pose_from_configuration
from agimus_controller_ros.parameters import OCPParameters


class OCPPoseRef:
    def __init__(
        self,
        rmodel: pin.Model,
        cmodel: pin.GeometryModel,
        params: OCPParameters,
        q_goal: np.array,
    ) -> None:
        """Class to define the OCP linked witha HPP generated trajectory.

        Args:
            rmodel (pin.Model): Pinocchio model of the robot.
            cmodel (pin.GeometryModel): Pinocchio geometry model of the robot. Must have been convexified for the collisions to work.
            params (OCPParameters) : parameters of the ocp.
        """

        self._rmodel = rmodel
        self._cmodel = cmodel
        self.params = params
        self._rdata = self._rmodel.createData()

        self._effector_frame_id = self._rmodel.getFrameId(params.effector_frame_name)
        self._weight_u_reg = params.control_weight
        self._weight_ee_placement = None
        self._weight_vel_reg = None
        self._weight_x_reg = params.state_weight
        self._collision_margin = 2e-2
        self.state = crocoddyl.StateMultibody(self._rmodel)
        self.actuation = crocoddyl.ActuationModelFull(self.state)
        self.nq = self._rmodel.nq  # Number of joints of the robot
        self.nv = self._rmodel.nv  # Dimension of the joint's speed vector of the robot
        self.x_plan = None
        self.a_plan = None
        self.u_plan = None
        self.running_models = None
        self.terminal_model = None
        self.solver = None
        self.next_node_time = None
        self.q_goal = q_goal
        self.x_goal = np.concatenate([q_goal, np.array([0.0] * self.nq)])
        self.des_pose = get_ee_pose_from_configuration(
            self._rmodel,
            self._rdata,
            self._effector_frame_id,
            np.array(q_goal),
        )

    def get_grav_compensation(
        self,
        q: np.ndarray,
    ) -> np.ndarray:
        """Return the reference of control u_plan that compensates gravity."""
        pin.computeGeneralizedGravity(
            self._rmodel,
            self._rdata,
            q,
        )
        return self._rdata.g.copy()

    def get_inverse_dynamic_control(self, x, a):
        """Return inverse dynamic control for a given state and acceleration."""
        return pin.rnea(self._rmodel, self._rdata, x[: self.nq], x[self.nq :], a).copy()

    def set_weights(
        self,
        weight_ee_placement: float,
        weight_x_reg: float,
        weight_u_reg: float,
        weight_vel_reg: float,
    ):
        """Set weights of the ocp.

        Args:
            weight_ee_placement (float): Weight of the placement of the end effector with regards to the target.
            weight_x_reg (float): Weight of the state regularization.
            weight_u_reg (float): Weight of the control regularization.
            weight_vel_reg (float): Weight of the velocity regularization.
        """
        self._weight_ee_placement = weight_ee_placement
        self._weight_x_reg = weight_x_reg
        self._weight_u_reg = weight_u_reg
        self._weight_vel_reg = weight_vel_reg

    def set_weight_ee_placement(self, weight_ee_placement: float):
        """Set end effector weight of the ocp.

        Args:
            weight_ee_placement (float): Weight of the placement of the end effector with regards to the target.
        """
        self._weight_ee_placement = weight_ee_placement

    def set_vel_weight(self, weight_vel_reg):
        self._weight_vel_reg = weight_vel_reg

    def set_control_weight(self, weight_u_reg):
        self._weight_u_reg = weight_u_reg

    def get_increasing_weight(self, time, max_weight):
        return max_weight * np.tanh(
            max(0.0, time)
            * np.arctanh(self.params.increasing_weights["percent"])
            / self.params.increasing_weights["time_reach_percent"]
        )

    def get_model(self, x_ref, u_ref, des_pose):
        running_cost_model = crocoddyl.CostModelSum(self.state)
        u_reg_cost = self.get_control_residual(u_ref)
        ugrav_reg_cost = self.get_control_grav_residual()
        x_reg_weights = np.array([1.0] * self.nq + [0.0] * self.nv)
        x_reg_cost = self.get_state_residual(x_ref, x_reg_weights)
        vel_reg_cost = self.get_velocity_residual()
        placement_reg_cost = self.get_placement_residual(des_pose)
        running_cost_model.addCost("uReg", u_reg_cost, 0)
        running_cost_model.addCost("ugravReg", ugrav_reg_cost, self._weight_u_reg)
        running_cost_model.addCost(
            "gripperPose", placement_reg_cost, self._weight_ee_placement
        )
        running_cost_model.addCost("velReg", vel_reg_cost, self._weight_vel_reg)
        running_cost_model.addCost("xReg", x_reg_cost, self._weight_x_reg)

        constraints = self.get_constraints()
        running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self.state, self.actuation, running_cost_model, constraints
        )
        running_DAM.armature = self.params.armature
        return crocoddyl.IntegratedActionModelEuler(running_DAM, self.params.dt)

    def get_terminal_model(self, x_ref, u_ref, des_pose):
        cost_model = crocoddyl.CostModelSum(self.state)
        u_reg_cost = self.get_control_residual(u_ref)
        ugrav_reg_cost = self.get_control_grav_residual()
        x_reg_cost = self.get_state_residual(x_ref)
        vel_reg_cost = self.get_velocity_residual()
        placement_reg_cost = self.get_placement_residual(des_pose)
        cost_model.addCost("uReg", u_reg_cost, 0)
        cost_model.addCost("ugravReg", ugrav_reg_cost, 0)
        cost_model.addCost("gripperPose", placement_reg_cost, 0)
        cost_model.addCost("velReg", vel_reg_cost, 0)
        cost_model.addCost("xReg", x_reg_cost, 0)
        running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self.state, self.actuation, cost_model
        )
        running_DAM.armature = self.params.armature
        return crocoddyl.IntegratedActionModelEuler(running_DAM, self.params.dt)

    def get_constraints(self):
        constraint_model_manager = crocoddyl.ConstraintModelManager(self.state, self.nq)
        if len(self._cmodel.collisionPairs) != 0:
            for col_idx in range(len(self._cmodel.collisionPairs)):
                collision_constraint = self._get_collision_constraint(
                    col_idx, self._collision_margin
                )
                # Adding the constraint to the constraint manager
                constraint_model_manager.addConstraint(
                    "col_term_" + str(col_idx), collision_constraint
                )
        return constraint_model_manager

    def _get_collision_constraint(
        self, col_idx: int, collision_margin: float
    ) -> "crocoddyl.ConstraintModelResidual":
        """Returns the collision constraint that will be in the constraint model manager.

        Args:
            col_idx (int): index of the collision pair.
            collision_margin (float): Lower bound of the constraint, ie the collision margin.

        Returns:
            _type_: _description_
        """
        obstacleDistanceResidual = ResidualDistanceCollision(
            self.state, 7, self._cmodel, col_idx
        )

        # Creating the inequality constraint
        constraint = crocoddyl.ConstraintModelResidual(
            self.state,
            obstacleDistanceResidual,
            np.array([collision_margin]),
            np.array([np.inf]),
        )
        return constraint

    def get_placement_residual(self, placement_ref):
        """Return placement residual with desired reference for end effector placement."""
        return crocoddyl.CostModelResidual(
            self.state,
            crocoddyl.ResidualModelFramePlacement(
                self.state, self._effector_frame_id, placement_ref
            ),
        )

    def get_velocity_residual(self):
        """Return velocity residual of desired joint."""
        vref = pin.Motion.Zero()
        return crocoddyl.CostModelResidual(
            self.state,
            crocoddyl.ResidualModelFrameVelocity(
                self.state,
                self._effector_frame_id,
                vref,
                pin.WORLD,
            ),
        )

    def get_control_residual(self, uref):
        """Return control residual with uref the control reference."""
        return crocoddyl.CostModelResidual(
            self.state, crocoddyl.ResidualModelControl(self.state, uref)
        )

    def get_control_grav_residual(self):
        """Return control residual with uref the control reference."""
        return crocoddyl.CostModelResidual(
            self.state, crocoddyl.ResidualModelControlGrav(self.state)
        )

    def get_state_residual(self, xref, x_reg_weights=None):
        """Return state residual with xref the state reference."""
        if x_reg_weights is None:
            return crocoddyl.CostModelResidual(
                self.state,
                crocoddyl.ResidualModelState(self.state, xref, self.actuation.nu),
            )
        else:
            return crocoddyl.CostModelResidual(
                self.state,
                crocoddyl.ActivationModelWeightedQuad(x_reg_weights),
                crocoddyl.ResidualModelState(self.state, xref, self.actuation.nu),
            )

    def update_cost(self, model, new_model, cost_name, update_weight=True):
        """Update model's cost reference and weight by copying new_model's cost."""
        model.differential.costs.costs[
            cost_name
        ].cost.residual.reference = new_model.differential.costs.costs[
            cost_name
        ].cost.residual.reference.copy()
        if update_weight:
            new_weight = new_model.differential.costs.costs[cost_name].weight
            model.differential.costs.costs[cost_name].weight = new_weight

    def update_weight(self, model, new_model, cost_name):
        new_weight = new_model.differential.costs.costs[cost_name].weight
        model.differential.costs.costs[cost_name].weight = new_weight

    def update_model(self, model, new_model, update_weight):
        """update model's costs by copying new_model's costs."""
        self.update_cost(model, new_model, "gripperPose", update_weight)
        self.update_cost(model, new_model, "velReg", update_weight)
        self.update_cost(model, new_model, "xReg", update_weight)
        self.update_weight(model, new_model, "ugravReg")

    def reset_ocp(self, x0, x_ref: np.ndarray, u_plan: np.ndarray, placement_ref):
        """Reset ocp problem using next reference in state and control."""
        self.solver.problem.x0 = x0
        u_grav = self.get_grav_compensation(x0[: self.nq])
        runningModels = list(self.solver.problem.runningModels)
        for node_idx in range(len(runningModels) - 1):
            self.update_model(
                runningModels[node_idx], runningModels[node_idx + 1], True
            )
        weight = self.get_increasing_weight(
            self.next_node_time - self.params.dt,
            self.params.increasing_weights["max"] / self.params.dt,
        )
        self.set_weight_ee_placement(weight)
        self.set_vel_weight(weight / 10)
        last_running_model = self.get_model(self.x_goal, u_grav, self.des_pose)
        self.update_model(runningModels[-1], last_running_model, True)
        terminal_model = self.get_terminal_model(x0, u_grav, self.des_pose)
        self.next_node_time += self.params.dt
        self.update_model(self.solver.problem.terminalModel, terminal_model, True)

    def set_planning_variables(self, x_plan: np.ndarray, a_plan: np.ndarray):
        self.x_plan = x_plan
        self.a_plan = a_plan
        u_grav = self.get_grav_compensation(x_plan[0, : self.nq])
        self.u_plan = np.array(list(u_grav) * (self.params.horizon_size - 1))
        self.u_plan = np.reshape(self.u_plan, (self.params.horizon_size - 1, 7))

    def build_ocp_from_plannif(self, x0):
        u_grav = self.u_plan[0, :]
        models = []
        for idx in range(self.params.horizon_size - 1):
            time = idx * self.params.dt - 0.2
            weight = self.get_increasing_weight(
                time, self.params.increasing_weights["max"] / self.params.dt
            )
            self.set_weight_ee_placement(weight)
            self.set_vel_weight(weight / 10)
            models.append(self.get_model(self.x_goal, u_grav, self.des_pose))
        terminal_model = self.get_terminal_model(x0, u_grav, self.des_pose)
        self.next_node_time = self.params.horizon_size * self.params.dt - 0.2

        return crocoddyl.ShootingProblem(x0, models, terminal_model)

    def run_solver(self, problem, xs_init, us_init):
        """
        Run FDDP or CSQP solver
        problem : crocoddyl ocp problem.
        xs_init : xs warm start.
        us_init : us warm start.
        max_iter : max number of iteration for the solver
        set_callback : activate solver callback
        """
        # Creating the solver for this OC problem, defining a logger
        solver = mim_solvers.SolverCSQP(problem)
        solver.use_filter_line_search = True
        solver.termination_tolerance = 1e-4
        solver.max_qp_iters = self.params.max_qp_iter
        solver.with_callbacks = self.params.activate_callback
        solver.solve(xs_init, us_init, self.params.max_iter)
        self.solver = solver
