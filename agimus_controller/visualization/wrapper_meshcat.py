import numpy as np
import hppfcl
import pinocchio as pin

import meshcat
import meshcat.geometry as g


RED = np.array([249, 136, 126, 125]) / 255
RED_FULL = np.array([249, 136, 126, 255]) / 255

GREEN = np.array([170, 236, 149, 125]) / 255
GREEN_FULL = np.array([170, 236, 149, 255]) / 255

BLUE = np.array([144, 169, 183, 125]) / 255
BLUE_FULL = np.array([144, 169, 183, 255]) / 255

YELLOW = np.array([1, 1, 0, 0.5])
YELLOW_FULL = np.array([1, 1, 0, 1.0])

BLACK = np.array([0, 0, 0, 0.5])
BLACK_FULL = np.array([0, 0, 0, 1.0])


def get_transform(T_: hppfcl.Transform3f):
    """Returns a np.ndarray instead of a pin.SE3 or a hppfcl.Transform3f

    Args:
        T_ (hppfcl.Transform3f): transformation to change into a np.ndarray. Can be a pin.SE3 as well

    Raises:
        NotADirectoryError: _description_

    Returns:
        _type_: _description_
    """
    T = np.eye(4)
    if isinstance(T_, hppfcl.Transform3f):
        T[:3, :3] = T_.getRotation()
        T[:3, 3] = T_.getTranslation()
    elif isinstance(T_, pin.SE3):
        T[:3, :3] = T_.rotation
        T[:3, 3] = T_.translation
    else:
        raise NotADirectoryError
    return T


class MeshcatWrapper:
    """Wrapper displaying a robot and a target in a meshcat server."""

    def __init__(self, grid=False, axes=False):
        """Wrapper displaying a robot and a target in a meshcat server.

        Args:
            grid (bool, optional): Boolean describing whether the grid will be displayed or not. Defaults to False.
            axes (bool, optional): Boolean describing whether the axes will be displayed or not. Defaults to False.
        """

        self._grid = grid
        self._axes = axes
        self.x0 = None
        self.last_x0 = None

    def visualize(
        self,
        TARGET=None,
        robot_model=None,
        robot_collision_model=None,
        robot_visual_model=None,
        robot_data=None,
        robot_collision_data=None,
        robot_visual_data=None,
    ):
        """Returns the visualiser, displaying the robot and the target if they are in input.

        Args:
            TARGET (pin.SE3, optional): pin.SE3 describing the position of the target. Defaults to None.
            robot_model (pin.Model, optional): pinocchio model of the robot. Defaults to None.
            robot_collision_model (pin.GeometryModel, optional): pinocchio collision model of the robot. Defaults to None.
            robot_visual_model (pin.GeometryModel, optional): pinocchio visual model of the robot. Defaults to None.

        Returns:
            tuple: viewer pinocchio and viewer meshcat.
        """
        # Creation of the visualizer,
        self.viewer = self.create_visualizer()

        if TARGET is not None:
            self._renderSphere("target", dim=5e-2, pose=TARGET)

        self._rmodel = robot_model
        self._cmodel = robot_collision_model
        self._vmodel = robot_visual_model

        self.viewer_pin = pin.visualize.MeshcatVisualizer(
            self._rmodel,
            collision_model=self._cmodel,
            visual_model=self._vmodel,
            data=robot_data,
            collision_data=robot_collision_data,
            visual_data=robot_visual_data,
        )
        self.viewer_pin.initViewer(
            viewer=meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
        )
        self.viewer_pin.loadViewerModel()
        self.viewer_pin.displayCollisions(True)

        return self.viewer_pin, self.viewer

    def create_visualizer(self):
        """Creation of an empty visualizer.

        Returns
        -------
        vis : Meshcat.Visualizer
            visualizer from meshcat
        """
        self.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
        self.viewer.delete()
        if not self._grid:
            self.viewer["/Grid"].set_property("visible", False)
        if not self._axes:
            self.viewer["/Axes"].set_property("visible", False)
        return self.viewer

    def _renderSphere(self, e_name: str, dim: np.ndarray, pose: pin.SE3, color=GREEN):
        """Displaying a sphere in a meshcat server.

        Parameters
        ----------
        e_name : str
            name of the object displayed
        color : np.ndarray, optional
            array describing the color of the target, by default np.array([1., 1., 1., 1.]) (ie white)
        """
        # Setting the object in the viewer
        self.viewer[e_name].set_object(g.Sphere(dim), self._meshcat_material(*color))
        T = get_transform(pose)

        # Applying the transformation to the object
        self.viewer[e_name].set_transform(T)

    def _meshcat_material(self, r, g, b, a):
        """Converting RGBA color to meshcat material.

        Args:
            r (int): color red
            g (int): color green
            b (int): color blue
            a (int): opacity

        Returns:
            material : meshcat.geometry.MeshPhongMaterial(). Material for meshcat
        """
        material = meshcat.geometry.MeshPhongMaterial()
        material.color = int(r * 255) * 256**2 + int(g * 255) * 256 + int(b * 255)
        material.opacity = a
        return material

    def get_geometry_dict(self):
        dict = {}
        for idx, obj in enumerate(self._cmodel.geometryObjects):
            dict[obj.name] = [obj, idx]
        return dict

    def remove_geom(self, name):
        self._cmodel.removeGeometryObject(name)
        self.refresh_meshcat()

    def set_geom_radius(self, idx, radius):
        self._cmodel.geometryObjects[idx].geometry.radius = radius
        self.refresh_meshcat()

    def set_geom_half_length(self, idx, halfLength):
        self._cmodel.geometryObjects[idx].geometry.halfLength = halfLength
        self.refresh_meshcat()

    def set_geom_placement(self, idx, translation):
        self._cmodel.geometryObjects[idx].placement.translation = translation
        self.refresh_meshcat()

    def set_geom_parent_frame(self, idx, frame_name):
        parentFrame = self._rmodel.getFrameId(frame_name)
        parentJoint = self._rmodel.getJointId(frame_name)
        self._cmodel.geometryObjects[idx].parentFrame = parentFrame
        self._cmodel.geometryObjects[idx].parentJoint = parentJoint
        self.refresh_meshcat()

    def set_geom_rotation_x(self, idx, angle):
        rot = self._cmodel.geometryObjects[idx].placement.rotation
        new_rot = np.matmul(rot, self.get_x_rot_matrix(angle))
        self._cmodel.geometryObjects[idx].placement.rotation = new_rot
        self.refresh_meshcat()

    def set_geom_rotation_y(self, idx, angle):
        rot = self._cmodel.geometryObjects[idx].placement.rotation
        new_rot = np.matmul(rot, self.get_y_rot_matrix(angle))
        self._cmodel.geometryObjects[idx].placement.rotation = new_rot
        self.refresh_meshcat()

    def set_geom_rotation_z(self, idx, angle):
        rot = self._cmodel.geometryObjects[idx].placement.rotation
        new_rot = np.matmul(rot, self.get_z_rot_matrix(angle))
        self._cmodel.geometryObjects[idx].placement.rotation = new_rot
        self.refresh_meshcat()

    def get_x_rot_matrix(self, angle):
        rot = np.zeros((3, 3))
        rot[0, 0] = 1
        rot[1, 1] = np.cos(angle)
        rot[2, 1] = np.sin(angle)
        rot[1, 2] = -np.sin(angle)
        rot[2, 2] = np.cos(angle)
        return rot

    def get_y_rot_matrix(self, angle):
        rot = np.zeros((3, 3))
        rot[1, 1] = 1
        rot[0, 0] = np.cos(angle)
        rot[2, 0] = np.sin(angle)
        rot[0, 2] = -np.sin(angle)
        rot[2, 2] = np.cos(angle)
        return rot

    def get_z_rot_matrix(self, angle):
        rot = np.zeros((3, 3))
        rot[2, 2] = 1
        rot[0, 0] = np.cos(angle)
        rot[1, 0] = np.sin(angle)
        rot[0, 1] = -np.sin(angle)
        rot[1, 1] = np.cos(angle)
        return rot

    def print_geometry_object_info(self, idx):
        print("name ", self._cmodel.geometryObjects[idx].name)
        print("placement \n", self._cmodel.geometryObjects[idx].placement)
        print("geometry type ", type(self._cmodel.geometryObjects[idx].geometry))
        if type(self._cmodel.geometryObjects[idx].geometry) is hppfcl.Capsule:
            print("radius ", self._cmodel.geometryObjects[idx].geometry.radius)
            print("halfLength ", self._cmodel.geometryObjects[idx].geometry.halfLength)

    def refresh_meshcat(self):
        self.vis = self.visualize(
            robot_model=self._rmodel,
            robot_visual_model=self._vmodel,
            robot_collision_model=self._cmodel,
        )
        self.viewer_pin.display(self.x0)

    def refresh_meshcat_at_last_pose(self):
        self.vis = self.visualize(
            robot_model=self._rmodel,
            robot_visual_model=self._vmodel,
            robot_collision_model=self._cmodel,
        )
        self.viewer_pin.display(self.last_x0)
