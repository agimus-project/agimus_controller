from __future__ import annotations
import numpy as np
from agimus_controller.utils.pin_utils import get_ee_pose_from_configuration


class MPC:
    """Create the MPC problem"""

    def __init__(
        self, ocp, x_plan: np.ndarray, a_plan: np.ndarray, rmodel, cmodel=None
    ):
        """Initiate the MPC problem.

        Args:
            ocp (OCP class): OCP describing the problem.
            x_plan (np.ndarray): State planification of HPP.
            a_plan (np.ndarray): Acceleration planification of HPP.
            rmodel (pin.Model): Pinocchio model of the robot
            cmodel (pin.CollisionModel): Pinocchio collision model.w
        """
        self.ocp = ocp
        self.whole_x_plan = x_plan
        self.whole_a_plan = a_plan
        self.rmodel = rmodel
        self.cmodel = cmodel
        self.nq = self.rmodel.nq
        self.nv = self.rmodel.nv
        self.nx = self.nq + self.nv
        self.croco_xs = None
        self.croco_us = None
        self.whole_traj_T = x_plan.shape[0]
        self.mpc_data = {}

    def get_next_state(self, x, problem):
        """Get state at the next step by doing a crocoddyl integration."""
        m = problem.runningModels[0]
        d = m.createData()
        m.calc(d, x, self.ocp.solver.us[0])
        return d.xnext.copy()

    def get_reference(self):
        model = self.ocp.solver.problem.runningModels[0]
        x_ref = model.differential.costs.costs["xReg"].cost.residual.reference.copy()
        p_ref = model.differential.costs.costs[
            "gripperPose"
        ].cost.residual.reference.translation.copy()
        u_ref = model.differential.costs.costs["uReg"].cost.residual.reference.copy()
        return x_ref, p_ref, u_ref

    def get_predictions(self):
        xs = np.array(self.ocp.solver.xs)
        us = np.array(self.ocp.solver.us)
        return xs, us

    def get_collision_residuals(self):
        constraints_residual_dict = self.ocp.solver.problem.runningDatas[
            0
        ].differential.constraints.constraints.todict()
        constraints_values = {}
        for constraint_key in constraints_residual_dict.keys():
            constraints_values[constraint_key] = list(
                constraints_residual_dict[constraint_key].residual.r
            )
        return constraints_values

    def simulate_mpc(self, save_predictions=False, node_idx_breakpoint=None):
        """Simulate mpc behavior using crocoddyl integration as a simulator."""
        T = self.ocp.params.horizon_size
        mpc_xs = np.zeros([self.whole_traj_T, 2 * self.nq])
        mpc_us = np.zeros([self.whole_traj_T - 1, self.nq])
        x0 = self.whole_x_plan[0, :]
        mpc_xs[0, :] = x0

        x_plan = self.whole_x_plan[:T, :]
        a_plan = self.whole_a_plan[:T, :]
        x, u0 = self.mpc_first_step(x_plan, a_plan, x0, T)
        mpc_xs[1, :] = x
        mpc_us[0, :] = u0
        next_node_idx = T

        if save_predictions:
            self.create_mpc_data(self.ocp.use_constraints)

        for idx in range(1, self.whole_traj_T - 1):
            x_plan = self.update_planning(x_plan, self.whole_x_plan[next_node_idx, :])
            a_plan = self.update_planning(a_plan, self.whole_a_plan[next_node_idx, :])
            placement_ref = get_ee_pose_from_configuration(
                self.ocp._rmodel,
                self.ocp._rdata,
                self.ocp._effector_frame_id,
                x_plan[-1, : self.nq],
            )
            x, u = self.mpc_step(x, x_plan[-1], a_plan[-1], placement_ref)
            if next_node_idx < self.whole_x_plan.shape[0] - 1:
                next_node_idx += 1
            mpc_xs[idx + 1, :] = x
            mpc_us[idx, :] = u

            if save_predictions:
                self.fill_predictions_and_refs_arrays(self.ocp.use_constraints)

            if idx == node_idx_breakpoint:
                breakpoint()
        self.croco_xs = mpc_xs
        self.croco_us = mpc_us
        if save_predictions:
            print("saving predictions in .npy files")
            np.save("mpc_data.npy", self.mpc_data)

    def update_planning(self, planning_vec, next_value):
        """Update numpy array by removing the first value and adding next_value at the end."""
        planning_vec = np.delete(planning_vec, 0, 0)
        return np.r_[planning_vec, next_value[np.newaxis, :]]

    def get_mpc_output(self):
        return self.ocp.solver.problem.x0, self.ocp.solver.us[0], self.ocp.solver.K[0]

    def mpc_first_step(self, x_plan, a_plan, x0, T):
        """Create crocoddyl problem from planning, run solver and get new state."""
        problem = self.ocp.build_ocp_from_plannif(x_plan, a_plan, x0)
        max_iter = self.ocp.params.max_iter
        self.ocp.params.max_iter = 1000
        self.ocp.run_solver(problem, list(x_plan), list(self.ocp.u_plan[: T - 1]))
        self.ocp.params.max_iter = max_iter
        x = self.get_next_state(x0, self.ocp.solver.problem)
        return x, self.ocp.solver.us[0]

    def mpc_step(self, x0, new_x_ref, new_a_ref, placement_ref):
        """Reset ocp, run solver and get new state."""
        u_ref_terminal_node = self.ocp.get_inverse_dynamic_control(new_x_ref, new_a_ref)
        self.ocp.reset_ocp(x0, new_x_ref, u_ref_terminal_node[: self.nq], placement_ref)
        xs_init = list(self.ocp.solver.xs[1:]) + [self.ocp.solver.xs[-1]]
        xs_init[0] = x0
        us_init = list(self.ocp.solver.us[1:]) + [self.ocp.solver.us[-1]]
        self.ocp.solver.problem.x0 = x0
        self.ocp.run_solver(self.ocp.solver.problem, xs_init, us_init)
        x0 = self.get_next_state(x0, self.ocp.solver.problem)
        return x0, self.ocp.solver.us[0]

    def create_mpc_data(self, use_constraints):
        xs, us = self.get_predictions()
        x_ref, p_ref, u_ref = self.get_reference()

        self.mpc_data["preds_xs"] = [xs]
        self.mpc_data["preds_us"] = [us]
        self.mpc_data["state_refs"] = [x_ref]
        self.mpc_data["translation_refs"] = [p_ref]
        self.mpc_data["control_refs"] = [u_ref]
        self.mpc_data["kkt_norm"] = [self.ocp.solver.KKT]

        if use_constraints:
            collision_residuals = self.get_collision_residuals()
            self.mpc_data["coll_residuals"] = collision_residuals

    def fill_predictions_and_refs_arrays(self, use_constraints):
        xs, us = self.get_predictions()
        x_ref, p_ref, u_ref = self.get_reference()
        self.mpc_data["preds_xs"].append(xs)

        self.mpc_data["preds_us"].append(us)
        self.mpc_data["state_refs"].append(x_ref)
        self.mpc_data["translation_refs"].append(p_ref)
        self.mpc_data["control_refs"].append(u_ref)
        self.mpc_data["kkt_norm"].append(self.ocp.solver.KKT)
        # if self.init_in_world_M_object is not None:
        #    self.mpc_data["init_in_world_M_object"] = self.init_in_world_M_object
        if use_constraints:
            collision_residuals = self.get_collision_residuals()
            for coll_residual_key in collision_residuals.keys():
                self.mpc_data["coll_residuals"][coll_residual_key] += (
                    collision_residuals[coll_residual_key]
                )
