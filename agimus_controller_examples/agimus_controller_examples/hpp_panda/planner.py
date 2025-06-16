import numpy as np
from hpp.corbaserver import Robot, ProblemSolver
from hpp.gepetto import ViewerFactory
import pinocchio as pin
from agimus_controller.factory.robot_model import RobotModels
from .scenes import Scene


def hack_for_ros2_support_in_hpp():
    import os

    if "ROS_PACKAGE_PATH" not in os.environ and "AMENT_PREFIX_PATH" in os.environ:
        os.environ["ROS_PACKAGE_PATH"] = ":".join(
            v + "/share" for v in os.environ["AMENT_PREFIX_PATH"].split(":")
        )


class Planner:
    def __init__(self, robot_models: RobotModels, scene: Scene, T: int) -> None:
        """Instatiate a motion planning class taking the pinocchio model and the geometry model.

        Args:
            rmodel (pin.Model): pinocchio model of the robot.
            cmodel (pin.GeometryModel): collision model of the robot
            scene (Scene): scene describing the environement.
            T (int): number of nodes describing the trajectory.
        """
        # Copy args.
        self._robot_models = robot_models
        self._scene = scene
        self._T = T

        # Models of the robot.
        self._rmodel = robot_models.robot_model
        self._cmodel = robot_models.collision_model

        # Visualizer
        self._v = None

    def _create_planning_scene(self, use_gepetto_gui):
        # Robot.urdfFilename = str(self._robot_models.params.robot_urdf)
        # Robot.srdfFilename = str(self._robot_models.params.srdf)

        # Client().problem.resetProblem()
        #
        hack_for_ros2_support_in_hpp()
        # package_location = "package://agimus_demo_05_pick_and_place"
        # urdf_string = process_xacro(package_location + "/urdf/demo.urdf.xacro")
        Robot.urdfString = self._robot_models._params.robot_urdf
        Robot.srdfString = ""
        # Client().problem.resetProblem()
        # robot = Robot("robot", "panda", rootJointType="anchor")
        robot = Robot("panda", rootJointType="anchor")
        self._ps = ProblemSolver(robot)
        # self._ps.loadObstacleFromUrdf(
        #    str(self._scene.urdf_model_path), self._scene._name_scene + "/"
        # )
        if use_gepetto_gui:
            vf = ViewerFactory(self._ps)
            vf.loadObstacleModel(
                str(self._scene.urdf_model_path), self._scene._name_scene, guiOnly=True
            )
        # for obstacle in self._cmodel.geometryObjects:
        #    if "obstacle" in obstacle.name:
        #        name = join(self._scene._name_scene, obstacle.name)
        #        scene_obs_pose = self._scene.obstacle_pose
        #        hpp_obs_pos = self._ps.getObstaclePosition(name)
        #        hpp_obs_pos[:3] += scene_obs_pose.translation[:3]
        #        self._ps.moveObstacle(name, hpp_obs_pos)
        #        if use_gepetto_gui:
        #            vf.moveObstacle(name, hpp_obs_pos, guiOnly=True)
        if use_gepetto_gui:
            self._v = vf.createViewer(collisionURDF=True)

    def setup_planner(self, q_init, q_goal, use_gepetto_gui=False):
        self._create_planning_scene(use_gepetto_gui)

        # Joints 8, and 9 are locked
        self._q_init = q_init + [0.0, 0.0]
        self._q_goal = [*q_goal, 0.03969, 0.0]
        q_init_list = self._q_init
        q_goal_list = self._q_goal
        self._ps.selectPathPlanner("BiRRT*")
        self._ps.setMaxIterPathPlanning(100)
        self._ps.setInitialConfig(q_init_list)
        self._ps.addGoalConfig(q_goal_list)

    def solve_and_optimize(self):
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
