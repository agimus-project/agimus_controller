from __future__ import annotations
import numpy as np
import time


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
        self.steps_duration = None

    def get_next_state(self, x, problem):
        """Get state at the next step by doing a crocoddyl integration."""
        m = problem.runningModels[0]
        d = m.createData()
        m.calc(d, x, self.ocp.solver.us[0])
        return d.xnext.copy()

    def get_reference(self):
        model = self.ocp.solver.problem.runningModels[0]
        x_ref = model.differential.costs.costs["xReg"].cost.residual.reference
        p_ref = model.differential.costs.costs[
            "gripperPose"
        ].cost.residual.reference.translation
        u_ref = model.differential.costs.costs["uReg"].cost.residual.reference
        return x_ref, p_ref, u_ref

    def simulate_mpc(self, T, save_predictions=False, node_idx_breakpoint=None):
        """Simulate mpc behavior using crocoddyl integration as a simulator."""
        self.steps_duration = np.zeros(self.whole_traj_T - 2)
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
            mpc_pred_xs = np.zeros([self.whole_traj_T, T, 2 * self.nq])
            mpc_pred_us = np.zeros([self.whole_traj_T, T - 1, self.nq])
            mpc_pred_xs[0, :, :] = np.array(self.ocp.solver.xs)
            mpc_pred_us[0, :, :] = np.array(self.ocp.solver.us)
            self.state_refs = np.zeros([self.whole_traj_T, 2 * self.nq])
            self.translation_refs = np.zeros([self.whole_traj_T, 3])
            self.control_refs = np.zeros([self.whole_traj_T, self.nq])
            x_ref, p_ref, u_ref = self.get_reference()
            self.state_refs[0, :] = x_ref
            self.translation_refs[0, :] = p_ref
            self.control_refs[0, :] = u_ref

        for idx in range(1, self.whole_traj_T - 1):
            x_plan = self.update_planning(x_plan, self.whole_x_plan[next_node_idx, :])
            a_plan = self.update_planning(a_plan, self.whole_a_plan[next_node_idx, :])
            start_mpc_step_time = time.time()
            x, u = self.mpc_step(x, x_plan[-1], a_plan[-1])
            self.steps_duration[idx - 1] = time.time() - start_mpc_step_time
            if next_node_idx < self.whole_x_plan.shape[0] - 1:
                next_node_idx += 1
            mpc_xs[idx + 1, :] = x
            mpc_us[idx, :] = u

            if save_predictions:
                mpc_pred_xs[idx, :, :] = np.array(self.ocp.solver.xs)
                mpc_pred_us[idx, :, :] = np.array(self.ocp.solver.us)
                x_ref, p_ref, u_ref = self.get_reference()
                self.state_refs[idx, :] = x_ref
                self.translation_refs[idx, :] = p_ref
                self.control_refs[idx, :] = u_ref

            if idx == node_idx_breakpoint:
                breakpoint()
        self.croco_xs = mpc_xs
        self.croco_us = mpc_us
        if save_predictions:
            print("saving")
            np.save("mpc_xs_sim.npy", mpc_pred_xs, allow_pickle=True)
            np.save("mpc_us_sim.npy", mpc_pred_us, allow_pickle=True)
            np.save("state_refs_sim.npy", self.state_refs)
            np.save("translation_refs_sim.npy", self.translation_refs)
            np.save("control_refs_sim.npy", self.control_refs)

    def update_planning(self, planning_vec, next_value):
        """Update numpy array by removing the first value and adding next_value at the end."""
        planning_vec = np.delete(planning_vec, 0, 0)
        return np.r_[planning_vec, next_value[np.newaxis, :]]

    def get_mpc_output(self):
        return self.ocp.solver.problem.x0, self.ocp.solver.us[0], self.ocp.solver.K[0]

    def mpc_first_step(self, x_plan, a_plan, x0, T):
        """Create crocoddyl problem from planning, run solver and get new state."""
        problem = self.ocp.build_ocp_from_plannif(x_plan, a_plan, x0)
        self.ocp.run_solver(problem, list(x_plan), list(self.ocp.u_plan[: T - 1]), 1000)
        x = self.get_next_state(x0, self.ocp.solver.problem)
        return x, self.ocp.solver.us[0]

    def mpc_step(self, x0, new_x_ref, new_a_ref):
        """Reset ocp, run solver and get new state."""
        u_ref_terminal_node = self.ocp.get_inverse_dynamic_control(new_x_ref, new_a_ref)
        self.ocp.reset_ocp(x0, new_x_ref, u_ref_terminal_node[: self.nq])
        xs_init = list(self.ocp.solver.xs[1:]) + [self.ocp.solver.xs[-1]]
        xs_init[0] = x0
        us_init = list(self.ocp.solver.us[1:]) + [self.ocp.solver.us[-1]]
        self.ocp.solver.problem.x0 = x0
        self.ocp.run_solver(self.ocp.solver.problem, xs_init, us_init, 1)
        x0 = self.get_next_state(x0, self.ocp.solver.problem)
        return x0, self.ocp.solver.us[0]
