import coal
from copy import deepcopy
from os import environ, pathsep
import unittest
from pathlib import Path
import example_robot_data as robex
import numpy as np
import pinocchio as pin
import yaml

# optional dependencies
try:
    import xacro

    XACRO_AVAILABLE = True
except ImportError as e:
    print(f"Error: {e}")  # Print the exact error
    XACRO_AVAILABLE = False
try:
    from ament_index_python.packages import (
        get_package_share_directory,
        PackageNotFoundError,
    )

    AMENT_AVAILABLE = True
except (ModuleNotFoundError, ImportError) as e:
    print(f"Error: {e}")  # Print the exact error
    AMENT_AVAILABLE = False
if AMENT_AVAILABLE:
    try:
        FRANKA_DESCRIPTION_PATH = get_package_share_directory("franka_description")
        environ["AMENT_PREFIX_PATH"] += pathsep + FRANKA_DESCRIPTION_PATH
        FRANKA_DESCRIPTION_AVAILABLE = True
    except (OSError, PackageNotFoundError) as e:
        print(f"Error: {e}")  # Print the exact error
        FRANKA_DESCRIPTION_AVAILABLE = False
else:
    FRANKA_DESCRIPTION_AVAILABLE = False


from agimus_controller.factory.robot_model import RobotModelParameters, RobotModels


class TestRobotModelParameters(unittest.TestCase):
    def setUp(self):
        robot = robex.load("panda")
        urdf_path = Path(robot.urdf)
        srdf_path = Path(robot.urdf.replace("urdf", "srdf"))
        self.valid_args = {
            "q0": np.array([0.0, 1.0, 2.0]),
            "urdf": urdf_path,
            "srdf": srdf_path,
            "urdf_meshes_dir": urdf_path.parent.parent.parent.parent.parent,
            "free_flyer": False,
            "moving_joint_names": [f"panda_joint{x}" for x in range(3, 8)],
            "collision_as_capsule": True,
            "self_collision": True,
        }
        reduced_nq = len(self.valid_args["moving_joint_names"])
        self.valid_args["armature"] = np.linspace(0.1, 0.9, reduced_nq)

    def test_valid_initialization(self):
        """Test that the dataclass initializes correctly with valid input."""

        self.valid_args["collision_as_capsule"] = True
        self.valid_args["self_collision"] = True

        params = RobotModelParameters(**deepcopy(self.valid_args))
        self.assertEqual(params.free_flyer, self.valid_args["free_flyer"])
        self.assertEqual(
            params.moving_joint_names, self.valid_args["moving_joint_names"]
        )
        self.assertEqual(params.urdf, self.valid_args["urdf"])
        self.assertEqual(params.srdf, self.valid_args["srdf"])
        self.assertEqual(params.urdf_meshes_dir, self.valid_args["urdf_meshes_dir"])
        self.assertTrue(params.collision_as_capsule)
        self.assertTrue(params.self_collision)
        np.testing.assert_array_equal(params.q0, self.valid_args["q0"])
        np.testing.assert_array_equal(params.armature, self.valid_args["armature"])

    def test_armature_default_value(self):
        """Test that the armature is set to default if not provided."""
        del self.valid_args["armature"]
        params = RobotModelParameters(**self.valid_args)
        np.testing.assert_array_equal(
            params.armature, np.zeros(len(self.valid_args["moving_joint_names"]))
        )

    def test_armature_mismatched_shape_raises_error(self):
        """Test that a mismatched armature raises a ValueError."""
        self.valid_args["q0"] = np.array([0.0, 1.0, 2.0])
        self.valid_args["armature"] = np.array([0.1, 0.2])  # Wrong shape
        with self.assertRaises(ValueError):
            RobotModelParameters(**self.valid_args)

    def test_invalid_urdf_raises_error(self):
        """Test that an invalid URDF path raises a ValueError."""
        for val in [Path("invalid_path"), ""]:
            self.valid_args["urdf"] = val
            with self.assertRaises(ValueError):
                RobotModelParameters(**self.valid_args)

    def test_invalid_srdf_type_raises_error(self):
        """Test that a non-string SRDF path raises a ValueError."""
        self.valid_args["srdf"] = Path("invalid_path")
        with self.assertRaises(ValueError):
            RobotModelParameters(**self.valid_args)

    def test_invalid_urdf_mesh_path_type_raises_error(self):
        """Test that a non-string SRDF path raises a ValueError."""
        self.valid_args["urdf_meshes_dir"] = Path("invalid_path")
        with self.assertRaises(ValueError):
            RobotModelParameters(**self.valid_args)


class TestRobotModelsAgainstExampleRobotData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        This method sets up the shared environment for all test cases in the class.
        """
        # Load the example robot model using example robot data to get the URDF path.
        robot = robex.load("panda")
        urdf_path = Path(robot.urdf)
        srdf_path = Path(robot.urdf.replace("urdf", "srdf"))
        urdf_meshes_dir = urdf_path.parent.parent.parent.parent.parent
        free_flyer = False
        q0 = np.zeros(robot.model.nq)
        moving_joint_names = [f"panda_joint{x}" for x in range(3, 8)]
        reduced_nq = len(moving_joint_names)
        cls.params = RobotModelParameters(
            q0=q0,
            free_flyer=free_flyer,
            moving_joint_names=moving_joint_names,
            urdf=urdf_path,
            srdf=srdf_path,
            urdf_meshes_dir=urdf_meshes_dir,
            collision_as_capsule=True,
            self_collision=True,
            armature=np.linspace(0.1, 0.9, reduced_nq),
        )

    def setUp(self):
        """
        This method ensures that a fresh RobotModelParameters and RobotModels instance
        are created for each test case.
        """
        self.params = deepcopy(self.params)
        self.robot_models = RobotModels(self.params)

    def test_no_meshes(self):
        self.params.urdf_meshes_dir = None
        # Clear paths where pinocchio searches for meshes
        if "ROS_PACKAGE_PATH" in environ:
            previous_ros_package_path = environ["ROS_PACKAGE_PATH"]
        else:
            previous_ros_package_path = ""
        if "AMENT_PREFIX_PATH" in environ:
            previous_ament_prefix_path = environ["AMENT_PREFIX_PATH"]
        else:
            previous_ament_prefix_path = ""
        environ["ROS_PACKAGE_PATH"] = ""
        environ["AMENT_PREFIX_PATH"] = ""
        with self.assertRaises(ValueError):
            RobotModels(self.params)
        environ["ROS_PACKAGE_PATH"] = previous_ros_package_path
        environ["AMENT_PREFIX_PATH"] = previous_ament_prefix_path

    def test_load_urdf_from_string(self):
        params = deepcopy(self.params)
        with open(Path(robex.load("panda").urdf), "r") as file:
            params.urdf = file.read().replace("\n", "")

        robot_models_str = RobotModels(params)
        self.assertEqual(
            robot_models_str.full_robot_model,
            self.robot_models.full_robot_model,
        )
        self.assertEqual(
            robot_models_str.robot_model,
            self.robot_models.robot_model,
        )
        np.testing.assert_array_equal(robot_models_str._q0, self.robot_models._q0)

        self.assertEqual(
            robot_models_str.collision_model.ngeoms,
            self.robot_models.collision_model.ngeoms,
        )
        self.assertEqual(
            robot_models_str.visual_model.ngeoms,
            self.robot_models.visual_model.ngeoms,
        )

    def test_initial_configuration(self):
        q0_test = self.params.q0.copy()
        np.testing.assert_array_equal(self.robot_models._q0, q0_test)
        self.params.q0 = np.array([], dtype=np.float64)
        self.robot_models.load_models()
        np.testing.assert_array_equal(self.robot_models._q0, q0_test)

    def test_load_models_populates_models(self):
        self.assertIsNotNone(self.robot_models.full_robot_model)
        self.assertIsNotNone(self.robot_models.visual_model)
        self.assertIsNotNone(self.robot_models.collision_model)

    def test_reduced_robot_model(self):
        self.assertTrue(
            self.robot_models.robot_model.nq == len(self.params.moving_joint_names)
        )

    def test_invalid_joint_name_raises_value_error(self):
        # Modify a fresh instance of parameters for this test
        self.params.moving_joint_names = ["InvalidJoint"]
        with self.assertRaises(ValueError):
            self.robot_models._lock_joints()

    def test_armature_property(self):
        np.testing.assert_array_equal(self.robot_models.armature, self.params.armature)

    def test_collision_pairs(self):
        """Checking that the collision model has collision pairs."""
        self.params.collision_as_capsule = False
        self.robot_models.load_models()
        self.assertEqual(
            len(self.robot_models.collision_model.collisionPairs), 44
        )  # Number of collision pairs in the panda model

    def test_rnea(self):
        """Checking that the RNEA method works."""
        q = np.zeros(self.robot_models.robot_model.nq)
        v = np.zeros(self.robot_models.robot_model.nv)
        a = np.zeros(self.robot_models.robot_model.nv)
        robot_data = self.robot_models.robot_model.createData()
        pin.rnea(self.robot_models.robot_model, robot_data, q, v, a)

    def test_franka_description_collision_models(self):
        geom_obj_names_test = [
            "panda_leftfinger_0",
            "panda_leftfinger_1",
            "panda_leftfinger_2",
            "panda_leftfinger_3",
            "panda_rightfinger_0",
            "panda_rightfinger_1",
            "panda_rightfinger_2",
            "panda_rightfinger_3",
        ]
        geom_obj_names = [
            geom_obj.name
            for geom_obj in self.robot_models.collision_model.geometryObjects
        ]
        for geom_obj_name, geom_obj_name_test in zip(
            geom_obj_names, geom_obj_names_test
        ):
            self.assertEqual(geom_obj_name, geom_obj_name_test)

        geom_obj_types_test = [
            coal.Box,
            coal.Box,
            coal.Box,
            coal.Box,
            coal.Box,
            coal.Box,
            coal.Box,
            coal.Box,
        ]
        geom_obj_types = [
            type(geom_obj.geometry)
            for geom_obj in self.robot_models.collision_model.geometryObjects
        ]
        self.assertEqual(geom_obj_types, geom_obj_types_test)


@unittest.skipIf(
    (not XACRO_AVAILABLE or not AMENT_AVAILABLE or not FRANKA_DESCRIPTION_AVAILABLE),
    f"Some dependencies amongst xacro (available ? {XACRO_AVAILABLE}) / "
    f"franka_description (available ? {FRANKA_DESCRIPTION_AVAILABLE}) / "
    f"ament_index_python (available ? {AMENT_AVAILABLE}) are not available",
)
class TestRobotModelsAgainstFrankaDescription(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        This method sets up the shared environment for all test cases in the class.
        """
        # Load the example robot model using example robot data to get the URDF path.
        franka_description_path = Path(
            get_package_share_directory("franka_description")
        )
        srdf_path = franka_description_path / "robots" / "fer" / "fer.srdf"
        xacro_path = str(
            franka_description_path / "robots" / "fer" / "fer.urdf.xacro",
        )
        params_path = str(
            Path(__file__).parent / "resources" / "agimus_controller_params.yaml"
        )
        with open(params_path, "r") as file:
            mpc_params = yaml.safe_load(file)["agimus_controller_node"][
                "ros__parameters"
            ]
        urdf_xml = xacro.process_file(
            xacro_path,
            mappings={"with_sc": "true"},
        ).toxml()
        # Hack for the moving joint name
        model = pin.buildModelFromXML(urdf_xml)
        cls.locked_joint_names = [
            jn
            for jn in model.names
            if jn not in ["universe"] + mpc_params["moving_joint_names"]
        ]
        cls.params = RobotModelParameters(
            q0=np.zeros(model.nq),
            free_flyer=False,
            moving_joint_names=mpc_params["moving_joint_names"],
            urdf=urdf_xml,
            srdf=srdf_path,
            collision_as_capsule=True,
            self_collision=True,
            urdf_meshes_dir=franka_description_path,
            armature=np.array(mpc_params["ocp"]["armature"]),
        )

    def setUp(self):
        """
        This method ensures that a fresh RobotModelParameters and RobotModels instance
        are created for each test case.
        """
        self.params = deepcopy(self.params)
        self.robot_models = RobotModels(self.params)

    def test_no_meshes(self):
        self.params.urdf_meshes_dir = None
        # Clear paths where pinocchio searches for meshes
        if "ROS_PACKAGE_PATH" in environ:
            previous_ros_package_path = environ["ROS_PACKAGE_PATH"]
        else:
            previous_ros_package_path = ""
        if "AMENT_PREFIX_PATH" in environ:
            previous_ament_prefix_path = environ["AMENT_PREFIX_PATH"]
        else:
            previous_ament_prefix_path = ""
        environ["ROS_PACKAGE_PATH"] = ""
        environ["AMENT_PREFIX_PATH"] = ""
        with self.assertRaises(ValueError):
            RobotModels(self.params)
        environ["ROS_PACKAGE_PATH"] = previous_ros_package_path
        environ["AMENT_PREFIX_PATH"] = previous_ament_prefix_path

    def test_initial_configuration(self):
        q0_test = self.params.q0.copy()
        np.testing.assert_array_equal(self.robot_models._q0, q0_test)
        self.params.q0 = np.array([], dtype=np.float64)
        self.robot_models.load_models()
        np.testing.assert_array_equal(self.robot_models._q0, q0_test)

    def test_load_models_populates_models(self):
        self.assertIsNotNone(self.robot_models.full_robot_model)
        self.assertIsNotNone(self.robot_models.visual_model)
        self.assertIsNotNone(self.robot_models.collision_model)

    def test_reduced_robot_model(self):
        self.assertTrue(
            self.robot_models.robot_model.nq == len(self.params.moving_joint_names)
        )

    def test_armature_property(self):
        np.testing.assert_array_equal(self.robot_models.armature, self.params.armature)

    def test_collision_pairs(self):
        """Checking that the collision model has collision pairs."""
        self.assertEqual(
            len(self.robot_models.collision_model.collisionPairs), 123
        )  # Number of collision pairs in the panda model

    def test_rnea(self):
        """Checking that the RNEA method works."""
        q = np.zeros(self.robot_models.robot_model.nq)
        v = np.zeros(self.robot_models.robot_model.nv)
        a = np.zeros(self.robot_models.robot_model.nv)
        robot_data = self.robot_models.robot_model.createData()
        pin.rnea(self.robot_models.robot_model, robot_data, q, v, a)

    def test_franka_description_collision_models(self):
        geom_obj_names_test = [
            "fer_leftfinger_0",
            "fer_leftfinger_1",
            "fer_leftfinger_2",
            "fer_leftfinger_3",
            "fer_rightfinger_0",
            "fer_rightfinger_1",
            "fer_rightfinger_2",
            "fer_rightfinger_3",
            "fer_hand_sc_capsule_0",
            "fer_hand_sc_capsule_1",
            "fer_link7_sc_capsule_0",
            "fer_link7_sc_capsule_1",
            "fer_link6_sc_capsule_0",
            "fer_link5_sc_capsule_0",
            "fer_link5_sc_capsule_1",
            "fer_link4_sc_capsule_0",
            "fer_link3_sc_capsule_0",
            "fer_link2_sc_capsule_0",
            "fer_link1_sc_capsule_0",
            "fer_link0_sc_capsule_0",
        ]
        geom_obj_names = [
            geom_obj.name
            for geom_obj in self.robot_models.collision_model.geometryObjects
        ]
        for geom_obj_name, geom_obj_name_test in zip(
            geom_obj_names, geom_obj_names_test
        ):
            self.assertEqual(geom_obj_name, geom_obj_name_test)

        geom_obj_types_test = [
            coal.Box,
            coal.Box,
            coal.Box,
            coal.Box,
            coal.Box,
            coal.Box,
            coal.Box,
            coal.Box,
            coal.Capsule,
            coal.Capsule,
            coal.Capsule,
            coal.Capsule,
            coal.Capsule,
            coal.Capsule,
            coal.Capsule,
            coal.Capsule,
            coal.Capsule,
            coal.Capsule,
            coal.Capsule,
            coal.Capsule,
        ]
        geom_obj_types = [
            type(geom_obj.geometry)
            for geom_obj in self.robot_models.collision_model.geometryObjects
        ]
        for geom_obj_type, geom_obj_type_test in zip(
            geom_obj_types, geom_obj_types_test
        ):
            self.assertEqual(geom_obj_type, geom_obj_type_test)


if __name__ == "__main__":
    unittest.main()
