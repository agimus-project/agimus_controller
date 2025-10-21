import itertools
import numpy as np
import numpy.typing as npt
import coal
import rclpy
import rclpy.duration
import sys
import argparse

import threading
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_anim

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from builtin_interfaces.msg import Duration as DurationMsg
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker
from agimus_msgs.msg import MpcDebug, MpcInput

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from agimus_controller.ocp.ocp_croco_generic import OCPCrocoGeneric
from agimus_controller.ocp_param_base import OCPParamsBaseCroco, DTFactorsNSeq
from agimus_controller.trajectory import (
    WeightedTrajectoryPoint,
)
from agimus_controller_ros.agimus_controller import RobotModelsMixin
from agimus_controller_ros.ros_utils import (
    get_params_from_node,
    mpc_msg_to_weighted_traj_point,
    transform_msg_to_se3,
    se3_to_pose_msg,
)

from linear_feedback_controller_msgs_py.numpy_conversions import matrix_msg_to_numpy
import pinocchio


def _capsule_as_markers(
    cap: coal.Cylinder, M: pinocchio.SE3, parent_name: str, **marker_args
) -> list[Marker]:
    marker_args.setdefault("action", Marker.ADD)
    marker_args.setdefault("color", ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0))
    marker_args.setdefault("lifetime", DurationMsg(sec=100000))
    marker_args.setdefault("header", Header(frame_id=parent_name))
    cylinder = Marker(**marker_args)
    cylinder.id = 0
    cylinder.type = Marker.CYLINDER
    cylinder.scale.x = cap.radius
    cylinder.scale.y = cap.radius
    cylinder.scale.z = 2 * cap.halfLength
    cylinder.pose = se3_to_pose_msg(M)

    M2 = pinocchio.SE3.Identity()

    sphere1 = Marker(**marker_args)
    sphere1.id = 1
    sphere1.type = Marker.SPHERE
    sphere1.scale.x = cap.radius
    sphere1.scale.y = cap.radius
    sphere1.scale.z = cap.radius
    M2.translation[2] = cap.halfLength
    sphere1.pose = se3_to_pose_msg(M * M2)

    sphere2 = Marker(**marker_args)
    sphere2.id = 2
    sphere2.type = Marker.SPHERE
    sphere2.scale.x = cap.radius
    sphere2.scale.y = cap.radius
    sphere2.scale.z = cap.radius
    M2.translation[2] = -cap.halfLength
    sphere2.pose = se3_to_pose_msg(M * M2)

    return cylinder, sphere1, sphere2


class MPCDebuggerNode(Node, RobotModelsMixin):
    """ROS node class to assist users of the AgimusController ROS node.

    Features:
    - publishes the current MPC prediction as markers.
    - publishes the current MPC references as markers.
    - publishes the capsules used for collision avoidance as markers.
    - show a bar plot of the items of the cost function.

    The markers can be visualized in RViz.
    """

    def __init__(
        self,
        frame_name: str,
        parent_frame_name: str,
        marker_size: float,
    ):
        """
        Args:
        - frame_name: name of the frame in the pinocchio Model
        - parent_frame_name: string passed to field `marker.header.frame_id` in the published messages.
        - marker_namespace: set the `marker.ns` field.
        - marker_size: set the `marker.scale` field.
        """
        super().__init__("mpc_debugger_node")
        self._frame_name = frame_name
        self._parent_frame_name = parent_frame_name
        self._marker_size = marker_size

        self._mpl_lock = threading.Lock()
        self._ocp_update_requested = threading.Event()
        self._mpl_data_ready = False
        self._stop_requested = False

        self.init_ros_robot_creation()
        self._get_agimus_controller_node_params()
        self._references: list[WeightedTrajectoryPoint] = list()

        self._mpc_input_sub = self.create_subscription(
            MpcInput,
            "mpc_input",
            self.mpc_input_callback,
            qos_profile=QoSProfile(
                depth=1000,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            ),
        )
        self._init_timer = self.create_timer(0.1, self.initialization_callback)

    def _get_agimus_controller_node_params(self):
        names = [
            "free_flyer",
            "ocp.dt_factor_n_seq.factors",
            "ocp.dt_factor_n_seq.n_steps",
            "collision_as_capsule",
            "self_collision",
            "ocp.armature",
            "ocp.definition_yaml_file",
            "ocp.dt",
            "ocp.horizon_size",
            # "ocp.max_iter",
            # "ocp.max_qp_iter",
            # "ocp.activate_callback",
        ]
        params = get_params_from_node(self, "agimus_controller_node", names)

        self._agimus_controller_node_params = dict(zip(names, params))

        self._robot_has_free_flyer = params[0].bool_value
        dt_factors = params[1].integer_array_value
        dt_n_steps = params[2].integer_array_value
        self._ocp_dt_factor_n_seq = DTFactorsNSeq(dt_factors, dt_n_steps)

        self._collision_as_capsule = params[3].bool_value
        self._self_collision = params[4].bool_value
        self._ocp_armature = np.array(params[5].double_array_value)

        self._horizon_indices = np.cumsum(
            sum(
                (
                    (factor,) * n_steps
                    for factor, n_steps in zip(dt_factors, dt_n_steps)
                ),
                (0,),
            )
        )

    def _initialize_capsules(self):
        capsules = []
        for i, go in enumerate(self.robot_models.collision_model.geometryObjects):
            if isinstance(go.geometry, coal.Capsule):
                frame = self.rmodel.frames[go.parentFrame]
                while (
                    frame.parentFrame > 0
                    and self.rmodel.frames[frame.parentFrame].type != pinocchio.JOINT
                ):
                    frame = self.rmodel.frames[frame.parentFrame]

                capsules.extend(
                    _capsule_as_markers(
                        go.geometry,
                        M=frame.placement.inverse() * go.placement,
                        parent_name=frame.name,
                        ns=go.name,
                    )
                )
        self._capsule_markers = MarkerArray(markers=capsules)
        if len(capsules) == 0:
            self.get_logger().warn(
                "No capsule found in the robot model. If you expect some, make sure you passed the same URDF as agimus_controller_node."
            )

        self._capsules_markers_pub = self.create_publisher(
            MarkerArray,
            "capsules_markers",
            qos_profile=QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
            ),
        )
        self._capsules_markers_pub.publish(self._capsule_markers)

    def _ocp_node_models(self):
        return itertools.chain(
            self._ocp._problem.runningModels, [self._ocp._problem.terminalModel]
        )

    def _ocp_node_datas(self):
        return itertools.chain(
            self._ocp._problem.runningDatas, [self._ocp._problem.terminalData]
        )

    def _initialize_ocp(self):
        params = self._agimus_controller_node_params
        ocp_params = OCPParamsBaseCroco(
            dt=params["ocp.dt"].double_value,
            dt_factor_n_seq=self._ocp_dt_factor_n_seq,
            horizon_size=params["ocp.horizon_size"].integer_value,
            solver_iters=10,
            callbacks=False,
            qp_iters=100,
            use_debug_data=False,
            n_threads=1,
        )

        yaml_file = params["ocp.definition_yaml_file"].string_value
        if yaml_file == "":
            yaml_file = OCPCrocoGeneric.get_default_yaml_file("ocp_goal_reaching.yaml")
        self.get_logger().info(f"Loading OCP definition file {yaml_file}")
        self._ocp = OCPCrocoGeneric(self.robot_models, ocp_params, yaml_file)

        if len(self._ocp.input_transforms) > 0:
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)

        cost_keys = {}
        self._ocp_cost_idx = []
        ocp_ndof = 0

        for data in self._ocp_node_datas():
            idx = []
            node_ndof = None
            for cost_item in data.differential.costs.costs:
                name = cost_item.key()
                idx.append(cost_keys.setdefault(name, len(cost_keys)))
                cost_data = cost_item.data()
                if node_ndof is None:
                    node_ndof = cost_data.Lx.shape[0] + cost_data.Lu.shape[0]
                else:
                    assert node_ndof == cost_data.Lx.shape[0] + cost_data.Lu.shape[0]
            ocp_ndof += node_ndof

            self._ocp_cost_idx.append(idx)

        self._ocp_cost_keys = sorted(cost_keys.keys(), key=cost_keys.__getitem__)
        self._ocp_cost_value = np.empty(
            (len(cost_keys), self._ocp._problem.T + 1), np.float64
        )
        self._ocp_cost_value[:] = np.nan

        self._ocp_cost_jacobian = np.empty((len(cost_keys), ocp_ndof), np.float64)
        self._ocp_cost_jacobian[:] = 0.0

    def _evaluate_ocp(
        self, states: npt.NDArray[np.float64], controls: npt.NDArray[np.float64]
    ):
        if self._ocp_update_requested.is_set():
            return

        # Step 1: Update references
        # Read transforms from TF. Note that doing do introduces a delay.
        now = self.get_clock().now()
        transforms = self._ocp.input_transforms
        for key in transforms:
            parent_frame, child_frame = key
            try:
                t = self._tf_buffer.lookup_transform(parent_frame, child_frame, now)
                M = transform_msg_to_se3(t.transform)
            except TransformException as ex:
                self.get_logger().info(
                    f"Could not transform {parent_frame} to {child_frame}: {ex}",
                    throttle_duration_sec=1.0,
                )
                M = None
            transforms[key] = M

        # Update the references from the horizon.
        references = [self._references[i] for i in self._horizon_indices]
        self._ocp.set_reference_weighted_trajectory(references)

        # Step 2: Build state and control and evaluate the OCP
        x = [np.asarray(state) for state in states]
        u = [np.asarray(control) for control in controls]
        problem = self._ocp._problem
        problem.calc(x, u)
        problem.calcDiff(x, u)

        # Step 3: retrieve the individual cost items.
        idof = 0
        with self._mpl_lock:
            for c, (cost_idx, model, data) in enumerate(
                zip(self._ocp_cost_idx, self._ocp_node_models(), self._ocp_node_datas())
            ):
                for r, cost_model, cost_data in zip(
                    cost_idx,
                    model.differential.costs.costs,
                    data.differential.costs.costs,
                ):
                    d = cost_data.data()
                    w = cost_model.data().weight
                    # CostModelSum adds the weight only when calculating the global cost so we need
                    # to take it into account here.
                    self._ocp_cost_value[r, c] = w * d.cost
                    cj = idof
                    self._ocp_cost_jacobian[r, cj : cj + d.Lx.shape[0]] = w * d.Lx
                    cj += d.Lx.shape[0]
                    self._ocp_cost_jacobian[r, cj : cj + d.Lu.shape[0]] = w * d.Lu
                idof += d.Lx.shape[0] + d.Lu.shape[0]

        assert idof == self._ocp_cost_jacobian.shape[1]

        self._ocp_update_requested.set()
        self._mpl_data_ready = True

    def _init_cost_plot(self):
        x = np.arange(self._ocp_cost_value.shape[1])  # the label locations
        width = 0.9 / self._ocp_cost_value.shape[0]  # the width of the bars
        multiplier = 0

        self._mpl_rects = []
        for ic, cost in enumerate(self._ocp_cost_value):
            offset = width * multiplier
            rects = self._mpl_ax_value.bar(
                x + offset, cost, width, label=self._ocp_cost_keys[ic]
            )
            self._mpl_rects.append(rects)
            multiplier += 1

        dt = self._agimus_controller_node_params["ocp.dt"].double_value
        self._mpl_ax_value.set_ylabel("cost value")
        self._mpl_ax_value.set_xlabel("OCP node (ms)")
        self._mpl_ax_value.set_xticks(
            x + 0.45, [f"{int(dt * 1000 * i)}" for i in self._horizon_indices]
        )
        self._mpl_ax_value.legend(loc="upper center")

        self._mpl_image = self._mpl_ax_jacobian.imshow(
            self._ocp_cost_jacobian,
            cmap="bwr",
            interpolation="none",
            aspect="auto",
            vmin=-1.0,
            vmax=1.0,
        )
        self._mpl_ax_jacobian.set_yticks(
            np.arange(len(self._ocp_cost_keys)), self._ocp_cost_keys
        )

        idof = 0
        xticks, xticks_label = list(), list()
        for c, (data) in enumerate(self._ocp_node_datas()):
            cost_data = next(iter(data.differential.costs.costs))
            d = cost_data.data()
            xticks.extend([idof, idof + d.Lx.shape[0]])
            xticks_label.extend([f"x{c}", f"u{c}"])
            idof += d.Lx.shape[0] + d.Lu.shape[0]

        self._mpl_ax_jacobian.set_xticks(xticks[0::2], xticks_label[0::2])
        self._mpl_ax_jacobian.set_xticks(xticks[1::2], xticks_label[1::2], minor=True)
        self._mpl_ax_jacobian.grid(axis="x", which="major", ls="-")
        self._mpl_ax_jacobian.grid(axis="x", which="minor", ls="--")
        self._mpl_ax_jacobian.set_title("Normalized Jacobian of the cost function")
        self._mpl_ax_jacobian.set_xlabel("OCP variables")
        plt.colorbar(self._mpl_image, ax=self._mpl_ax_jacobian, location="bottom")

    def _update_cost_plot(self, _):
        """Function for for adding data to axis.

        Args:
            _ : Dummy variable that is required for matplotlib animation.

        Returns:
            Axes object for matplotlib
        """
        if not self._ocp_update_requested.is_set():
            return

        if self._stop_requested:
            self._mpl_ani.pause()
            return self._mpl_ax_value

        # lock thread
        with self._mpl_lock:
            if not hasattr(self, "_mpl_rects"):
                self._init_cost_plot()
            else:
                for rects, costs in zip(self._mpl_rects, self._ocp_cost_value):
                    for rect, cost in zip(rects, costs):
                        rect.set_height(cost)
                self._mpl_ax_value.set_ylim(ymax=np.nanmax(self._ocp_cost_value))

                s = np.max(np.abs(self._ocp_cost_jacobian))
                np.save("/tmp/data.npy", self._ocp_cost_jacobian)
                self._mpl_image.set_data(self._ocp_cost_jacobian / s)

            self._ocp_update_requested.clear()
            return self._mpl_ax_value, self._mpl_ax_jacobian

    def plot_cost(self, update_interval_ms=250):
        """Function for initializing and showing matplotlib animation."""
        self._mpl_fig, (self._mpl_ax_value, self._mpl_ax_jacobian) = plt.subplots(
            nrows=2,
            ncols=1,
            layout="tight",
        )

        self._mpl_ani = mpl_anim.FuncAnimation(
            self._mpl_fig,
            self._update_cost_plot,
            interval=update_interval_ms,
        )
        plt.show()

    def initialization_callback(self):
        if not self.ros_robot_ready():
            self.get_logger().warn(
                "Waiting for robot descriptions...",
                throttle_duration_sec=5.0,
            )
            return

        self.destroy_timer(self._init_timer)

        self.get_logger().info("create robot...")
        self.create_robot_models(
            free_flyer=self._robot_has_free_flyer,
            collision_as_capsule=self._collision_as_capsule,
            self_collision=self._self_collision,
            armature=self._ocp_armature,
        )
        frame_name_ok = self.rmodel.existFrame(self._frame_name)
        assert frame_name_ok, f"Frame {self._frame_name} could not be found."
        self.rdata = self.rmodel.createData()
        self._fid = self.rmodel.getFrameId(self._frame_name)

        self._initialize_capsules()
        self._initialize_ocp()

        self._pred_marker_array = MarkerArray(
            markers=self._create_marker_array(
                namespace="states_predictions",
                size=len(self._horizon_indices),
                rgba0=[1.0, 0.0, 0.0, 1.0],
                rgba1=[0.5, 1.0, 0.0, 0.2],
            )
        )
        self._ref_marker_array = MarkerArray(
            markers=self._create_marker_array(
                namespace="states_references",
                size=len(self._horizon_indices),
                rgba0=[0.0, 0.0, 1.0, 1.0],
                rgba1=[0.0, 1.0, 0.5, 0.2],
            )
        )
        self._ref_marker_array_pose = MarkerArray(
            markers=self._create_marker_array(
                namespace="states_references_pose",
                size=len(self._horizon_indices),
                rgba0=[0.0, 1.0, 0.0, 1.0],
                rgba1=[0.0, 0.5, 1.0, 0.2],
            )
        )
        self._mpc_debug_sub = self.create_subscription(
            MpcDebug,
            "mpc_debug",
            self.mpc_debug_to_markers,
            qos_profile=QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            ),
        )
        self._mpc_ref_markers_pub = self.create_publisher(
            MarkerArray,
            "mpc_states_reference_markers",
            qos_profile=QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )
        self._mpc_ref_markers_pose_pub = self.create_publisher(
            MarkerArray,
            "mpc_states_reference_markers_pose",
            qos_profile=QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )
        self._mpc_pred_markers_pub = self.create_publisher(
            MarkerArray,
            "mpc_states_prediction_markers",
            qos_profile=QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )
        self.get_logger().info("init done")

    def mpc_input_callback(self, msg: MpcInput):
        w_traj_point = mpc_msg_to_weighted_traj_point(
            msg, time_ns=self.get_clock().now().nanoseconds
        )
        if len(self._references) > 0:
            assert w_traj_point.point.id == self._references[-1].point.id + 1, (
                "MPC input ids are expected to be a sequence of consecutive increasing integers."
            )
        self._references.append(w_traj_point)

    def _create_marker_array(
        self, namespace: str, size: int, rgba0, rgba1
    ) -> list[Marker]:
        markers = []
        for i in range(size):
            marker = Marker()
            marker.header.frame_id = self._parent_frame_name
            # The MPC debug message does not have a stamp so
            # it is not possible to correctly set
            # marker.header.stamp
            marker.ns = namespace
            marker.id = i

            marker.type = Marker.SPHERE

            marker.action = Marker.ADD

            marker.scale.x = self._marker_size
            marker.scale.y = self._marker_size
            marker.scale.z = self._marker_size

            # Interpolate from rgba0 to rgba1
            r = i / (size - 1)
            marker.color.r = rgba0[0] + r * (rgba1[0] - rgba0[0])
            marker.color.g = rgba0[1] + r * (rgba1[1] - rgba0[1])
            marker.color.b = rgba0[2] + r * (rgba1[2] - rgba0[2])
            marker.color.a = rgba0[3] + r * (rgba1[3] - rgba0[3])

            marker.lifetime = DurationMsg(sec=1)
            markers.append(marker)
        return markers

    def _remove_old_references(self, id: int):
        while self._references[0].point.id < id:
            self._references.pop(0)

    def mpc_debug_to_markers(self, msg: MpcDebug):
        states = matrix_msg_to_numpy(msg.states_predictions)
        assert states.shape[0] == len(self._pred_marker_array.markers), (
            f"{states.shape[0]} != {len(self._pred_marker_array.markers)}"
        )
        nq = self.rmodel.nq
        for state, marker in zip(states, self._pred_marker_array.markers):
            pinocchio.forwardKinematics(self.rmodel, self.rdata, state[:nq])
            M = pinocchio.updateFramePlacement(self.rmodel, self.rdata, self._fid)
            marker.pose = se3_to_pose_msg(M)

        self._remove_old_references(msg.trajectory_point_id)
        if msg.trajectory_point_id == self._references[0].point.id:
            assert len(self._horizon_indices) == len(self._ref_marker_array.markers), (
                f"{len(self._horizon_indices)} != {len(self._ref_marker_array.markers)}"
            )
            assert len(self._horizon_indices) == len(self._ref_marker_array_pose.markers), (
                f"{len(self._horizon_indices)} != {len(self._ref_marker_array_pose.markers)}"
            )

            for i, marker, marker_pose in zip(self._horizon_indices,
                                              self._ref_marker_array.markers,
                                              self._ref_marker_array_pose.markers):
                if i >= len(self._references):
                    self.get_logger().warn(
                        f"Not enough references. Nb ref is {len(self._references)}. Need {self._horizon_indices[-1] + 1}",
                        throttle_duration_sec=1.0,
                    )
                    break
                state = self._references[i].point.robot_configuration
                assert len(state) == nq, f"{len(state)} == {nq}"
                pinocchio.forwardKinematics(self.rmodel, self.rdata, np.asarray(state))
                M = pinocchio.updateFramePlacement(self.rmodel, self.rdata, self._fid)
                marker.pose = se3_to_pose_msg(M)

                pose = list(self._references[i].point.end_effector_poses.values())[0]
                # TODO how to use the correct ee frame from dict?
                marker_pose.pose = se3_to_pose_msg(pose)

            if len(self._references) > self._horizon_indices[-1]:
                controls = matrix_msg_to_numpy(msg.control_predictions)
                self._evaluate_ocp(states, controls)
            self._mpc_ref_markers_pub.publish(self._ref_marker_array)
            self._mpc_ref_markers_pose_pub.publish(self._ref_marker_array_pose)
        else:
            self.get_logger().warn(
                f"First ref id: {self._references[0].point.id}. Msg id: {msg.trajectory_reference_id}",
                throttle_duration_sec=1.0,
            )

        self._mpc_pred_markers_pub.publish(self._pred_marker_array)
        if msg.trajectory_point_id % 10 == 0:
            self._capsules_markers_pub.publish(self._capsule_markers)


def main(args=None):
    # Initialize rclpy first to handle ROS 2 arguments
    rclpy.init(args=args)

    # Filter out ROS 2-specific arguments before passing to argparse
    filtered_args = rclpy.utilities.remove_ros_args(args)

    # Use argparse to parse the remaining arguments
    parser = argparse.ArgumentParser(
        "mpc_debugger_node",
        description="This node transforms the MPC debug data into a marker array that can be visualized in RViz.",
    )
    parser.add_argument(
        "--frame",
        type=str,
        required=True,
        help="name of the frame in the pinocchio Model",
    )
    parser.add_argument(
        "--parent-frame",
        type=str,
        default="world",
        help="string passed to field `marker.header.frame_id` in the published messages.",
    )
    parser.add_argument(
        "--marker-size", type=float, default=0.01, help="set the `marker.scale` field."
    )
    parser.add_argument(
        "--cost-plot",
        action=argparse.BooleanOptionalAction,
        help="create a plot of the cost function.",
    )

    arguments = parser.parse_args(filtered_args[1:])  # Skip the script name

    node = MPCDebuggerNode(
        frame_name=arguments.frame,
        parent_frame_name=arguments.parent_frame,
        marker_size=arguments.marker_size,
    )

    if arguments.cost_plot:
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(node)
        thread = threading.Thread(target=executor.spin, daemon=True)
    try:
        if arguments.cost_plot:
            thread.start()
            node.plot_cost()
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        if arguments.cost_plot:
            # TODO I don't know how to request the matplotlib figure to be closed.
            # My attempt have not been successful.
            node._stop_requested = True
            plt.close(node._mpl_fig)
            executor.shutdown()
            thread.join()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
