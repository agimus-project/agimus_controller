"""Implement PLanner."""

from os.path import join
import numpy as np
from hpp.corbaserver import Client, Robot, ProblemSolver
from hpp.gepetto import ViewerFactory
import pinocchio as pin
from ..robot_model.robot_model import RobotModel
from .scenes import Scene


class Planner:
    """Plan a trajectory from a goal to end configuration."""

    def __init__(self, robot_model: RobotModel, scene: Scene, T: int) -> None:
        """Instatiate a motion planning class taking the pinocchio model and the geometry model.

        Args:
            robot_model (RobotModel): model of the robot.
            scene (Scene): scene describing the environement.
            T (int): number of nodes describing the trajectory.

        """
        # Copy args.
        self._robot_model = robot_model
        self._scene = scene
        self._T = T

        # Models of the robot.
        self._robot_model_params = self._robot_model.get_model_parameters()
        self._rmodel = self._robot_model.get_reduced_robot_model()
        self._cmodel = self._robot_model.get_reduced_collision_model()
        self._end_effector_id = self._rmodel.getFrameId(
            self._robot_model_params.ee_frame_name
        )

        # Visualizer
        self._v = None

    def _create_planning_scene(self, use_gepetto_gui):
        """Create a planning scene in HPP."""
        Robot.urdfFilename = str(self._robot_model_params.urdf)
        Robot.srdfFilename = str(self._robot_model_params.srdf)

        Client().problem.resetProblem()

        robot = Robot("panda", rootJointType="anchor")
        self._ps = ProblemSolver(robot)
        self._ps.loadObstacleFromUrdf(
            str(self._scene.urdf_model_path), self._scene._name_scene + "/"
        )
        if use_gepetto_gui:
            vf = ViewerFactory(self._ps)
            vf.loadObstacleModel(
                str(self._scene.urdf_model_path), self._scene._name_scene, guiOnly=True
            )
        for obstacle in self._cmodel.geometryObjects:
            if "obstacle" in obstacle.name:
                name = join(self._scene._name_scene, obstacle.name)
                scene_obs_pose = self._scene.obstacle_pose
                hpp_obs_pos = self._ps.getObstaclePosition(name)
                hpp_obs_pos[:3] += scene_obs_pose.translation[:3]
                self._ps.moveObstacle(name, hpp_obs_pos)
                if use_gepetto_gui:
                    vf.moveObstacle(name, hpp_obs_pos, guiOnly=True)
        if use_gepetto_gui:
            self._v = vf.createViewer(collisionURDF=True)

    def setup_planner(
        self, q_init: np.ndarray, q_goal: np.ndarray, use_gepetto_gui: bool
    ):
        """Set up the planner.

        Args:
            q_init (np.ndarray): Starting configuration.
            q_goal (np.ndarray): Goal configuration.
            use_gepetto_gui (bool): Do we export the trajectory in a gui?

        """
        self._create_planning_scene(use_gepetto_gui)

        # Joints 8, and 9 are locked
        self._q_init = [*q_init, 0.03969, 0.03969]
        self._q_goal = [*q_goal, 0.03969, 0.03969]
        q_init_list = self._q_init
        q_goal_list = self._q_goal
        if use_gepetto_gui:
            self._v(q_init_list)
        self._ps.selectPathPlanner("BiRRT*")
        self._ps.setMaxIterPathPlanning(100)
        self._ps.setInitialConfig(q_init_list)
        self._ps.addGoalConfig(q_goal_list)

    def solve_and_optimize(self):
        """Solve the path planning problem and optimize the path."""
        self._ps.setRandomSeed(1)
        self._ps.solve()
        self._ps.getAvailable("pathoptimizer")
        self._ps.selectPathValidation("Dichotomy", 0)
        self._ps.addPathOptimizer("SimpleTimeParameterization")
        self._ps.setParameter("SimpleTimeParameterization/maxAcceleration", 1.0)
        self._ps.setParameter("SimpleTimeParameterization/order", 2)
        self._ps.setParameter("SimpleTimeParameterization/safety", 0.9)
        self._ps.addPathOptimizer("RandomShortcut")
        self._ps.solve()
        path_length = self._ps.pathLength(2)
        X = [
            self._ps.configAtParam(0, i * path_length / self._T)[:7]
            for i in range(self._T)
        ]
        return self._q_init, self._q_goal, np.array(X)

    def _generate_feasible_configurations(self):
        """Genereate a random feasible configuration of the robot.

        Returns:
            q np.ndarray: configuration vector of the robot.

        """
        q = pin.randomConfiguration(self._rmodel)
        while self._check_collisions(q):
            q = pin.randomConfiguration(self._rmodel)
        return q

    def _generate_feasible_configurations_array(self):
        """Generate configuration that are feasible kinematically."""
        col = True
        while col:
            q = np.zeros(self._rmodel.nq)
            for i, qi in enumerate(q):
                lb = self._rmodel.lowerPositionLimit[i]
                ub = self._rmodel.upperPositionLimit[i]
                margin = 0.2 * abs(ub - lb) / 2
                q[i] = np.random.uniform(
                    self._rmodel.lowerPositionLimit[i] + margin,
                    self._rmodel.upperPositionLimit[i] - margin,
                    1,
                )
            col = self._check_collisions(q)
        return q

    def _check_collisions(self, q: np.ndarray):
        """Check the collisions for a given configuration array.

        Args:
            q (np.ndarray): configuration array
        Returns:
            col (bool): True if no collision

        """
        rdata = self._rmodel.createData()
        cdata = self._cmodel.createData()
        col = pin.computeCollisions(self._rmodel, rdata, self._cmodel, cdata, q, True)
        return col

    def _inverse_kinematics(
        self, target_pose, initial_guess=None, max_iters=1000, tol=1e-6
    ):
        """Solve the inverse kinematics problem for a given robot and target pose.

        Args:
        target_pose (pin.SE3): Desired end-effector pose (as a pin.SE3 object)
        initial_guess (np.ndarray): Initial guess for the joint configuration (optional)
        max_iters (int): Maximum number of iterations
        tol (float): Tolerance for convergence

        Returns:
        q_sol (np.ndarray): Joint configuration that achieves the target pose

        """
        rdata = self._rmodel.createData()

        if initial_guess is None:
            q = pin.neutral(
                self._rmodel
            )  # Use the neutral configuration as the initial guess
        else:
            q = initial_guess

        for i in range(max_iters):
            # Compute the current end-effector pose
            pin.forwardKinematics(self._rmodel, rdata, q)
            pin.updateFramePlacements(self._rmodel, rdata)
            current_pose = rdata.oMf[self._end_effector_id]

            # Compute the error between current and target poses
            error = pin.log6(current_pose.inverse() * target_pose).vector
            if np.linalg.norm(error) < tol:
                print(f"Converged in {i} iterations.")
                return q

            # Compute the Jacobian of the end effector
            J = pin.computeFrameJacobian(self._rmodel, rdata, q, self._end_effector_id)

            # Compute the change in joint configuration using the pseudo-inverse of the Jacobian
            dq = np.linalg.pinv(J) @ error

            # Update the joint configuration
            q = pin.integrate(self._rmodel, q, dq)

        raise RuntimeError("Inverse kinematics did not converge")
