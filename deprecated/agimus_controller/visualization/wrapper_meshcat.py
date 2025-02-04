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
