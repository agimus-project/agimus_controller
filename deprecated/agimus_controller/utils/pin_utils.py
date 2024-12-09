from typing import Tuple

import hppfcl
import numpy as np
import warnings
import pinocchio as pin


# Rotate placement
def rotate(se3_placement, rpy=[0.0, 0.0, 0.0]):
    """
    Rotates se3_placement.rotation by rpy (LOCAL)
     input :
        se3_placement : pin.SE3
        rpy           : RPY orientation in LOCAL frame
                        RPY
    """
    se3_placement_rotated = se3_placement.copy()
    R = pin.rpy.rpyToMatrix(rpy[0], rpy[1], rpy[2])
    se3_placement_rotated.rotation = se3_placement_rotated.rotation.copy().dot(R)
    return se3_placement_rotated


# Get frame position
def get_p(q, pin_robot, id_endeff):
    """
    Returns end-effector positions given q trajectory
        q         : joint positions
        robot     : pinocchio wrapper
        id_endeff : id of EE frame
    """
    return get_p_(q, pin_robot.model, id_endeff)


def get_p_(q, model, id_endeff):
    """
    Returns end-effector positions given q trajectory
        q         : joint positions
        model     : pinocchio model
        id_endeff : id of EE frame
    """

    data = model.createData()
    if type(q) is np.ndarray and len(q.shape) == 1:
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        p = data.oMf[id_endeff].translation.T
    else:
        N = np.shape(q)[0]
        p = np.empty((N, 3))
        for i in range(N):
            pin.forwardKinematics(model, data, q[i])
            pin.updateFramePlacements(model, data)
            p[i, :] = data.oMf[id_endeff].translation.T
    return p


# Get frame linear velocity
def get_v(q, dq, pin_robot, id_endeff, ref=pin.LOCAL):
    """
    Returns end-effector velocities given q,dq trajectory
        q         : joint positions
        dq        : joint velocities
        pin_robot : pinocchio wrapper
        id_endeff : id of EE frame
    """
    return get_v_(q, dq, pin_robot.model, id_endeff, ref)


def get_v_(q, dq, model, id_endeff, ref=pin.LOCAL):
    """
    Returns end-effector velocities given q,dq trajectory
        q         : joint positions
        dq        : joint velocities
        model     : pinocchio model
        id_endeff : id of EE frame
    """
    data = model.createData()
    if len(q) != len(dq):
        print("q and dq must have the same size !")
    if type(q) is np.ndarray and len(q.shape) == 1:
        # J = pin.computeFrameJacobian(model, data, q, id_endeff)
        # v = J.dot(dq)[:3]
        pin.forwardKinematics(model, data, q, dq)
        spatial_vel = pin.getFrameVelocity(model, data, id_endeff, ref)
        v = spatial_vel.linear
    else:
        N = np.shape(q)[0]
        v = np.empty((N, 3))
        for i in range(N):
            # J = pin.computeFrameJacobian(model, data, q[i,:], id_endeff)
            # v[i,:] = J.dot(dq[i])[:3]
            pin.forwardKinematics(model, data, q[i], dq[i])
            spatial_vel = pin.getFrameVelocity(model, data, id_endeff, ref)
            v[i, :] = spatial_vel.linear
    return v


# Get frame orientation (rotation)
def get_R(q, pin_robot, id_endeff):
    """
    Returns end-effector rotation matrices given q trajectory
        q         : joint positions
        pin_robot : pinocchio wrapper
        id_endeff : id of EE frame
    """
    return get_R_(q, pin_robot.model, id_endeff)


def get_R_(q, model, id_endeff):
    """
    Returns end-effector rotation matrices given q trajectory
        q         : joint positions
        model     : pinocchio model
        id_endeff : id of EE frame
    Output : single 3x3 array (or list of 3x3 arrays)
    """
    data = model.createData()
    if type(q) is np.ndarray and len(q.shape) == 1:
        pin.framesForwardKinematics(model, data, q)
        R = data.oMf[id_endeff].rotation.copy()
    else:
        N = np.shape(q)[0]
        R = []
        for i in range(N):
            pin.framesForwardKinematics(model, data, q[i])
            R.append(data.oMf[id_endeff].rotation.copy())
    return R


# Get frame orientation (RPY)
def get_rpy(q, pin_robot, id_endeff):
    """
    Returns RPY angles of end-effector frame given q trajectory
        q         : joint positions
        model     : pinocchio wrapper
        id_endeff : id of EE frame
    """
    return get_rpy_(q, pin_robot.model, id_endeff)


def get_rpy_(q, model, id_endeff):
    """
    Returns RPY angles of end-effector frame given q trajectory
        q         : joint positions
        model     : pinocchio model
        id_endeff : id of EE frame
    """
    R = get_R_(q, model, id_endeff)
    if isinstance(R, list):
        N = np.shape(q)[0]
        rpy = np.empty((N, 3))
        for i in range(N):
            rpy[i, :] = pin.rpy.matrixToRpy(R[i])  # %(2*np.pi)
    else:
        rpy = pin.rpy.matrixToRpy(R)  # %(2*np.pi)
    return rpy


# Get frame angular velocity
def get_w(q, dq, pin_robot, id_endeff, ref=pin.LOCAL):
    """
    Returns end-effector angular velocity given q,dq trajectory
        q         : joint positions
        dq        : joint velocities
        pin_robot : pinocchio wrapper
        id_endeff : id of EE frame
    """
    return get_w_(q, dq, pin_robot.model, id_endeff, ref)


def get_w_(q, dq, model, id_endeff, ref=pin.LOCAL):
    """
    Returns end-effector  angular velocity given q,dq trajectory
        q         : joint positions
        dq        : joint velocities
        pin_robot : pinocchio wrapper
        id_endeff : id of EE frame
    """
    data = model.createData()
    if len(q) != len(dq):
        print("q and dq must have the same size !")
    if type(q) is np.ndarray and len(q.shape) == 1:
        pin.forwardKinematics(model, data, q, dq)
        spatial_vel = pin.getFrameVelocity(model, data, id_endeff, ref)
        w = spatial_vel.angular
    else:
        N = np.shape(q)[0]
        w = np.empty((N, 3))
        for i in range(N):
            pin.forwardKinematics(model, data, q[i], dq[i])
            spatial_vel = pin.getFrameVelocity(model, data, id_endeff, ref)
            w[i, :] = spatial_vel.angular
    return w


# Get gravity joint torque
def get_u_grav(q, model, armature):
    """
    Return gravity torque at q
    """
    data = model.createData()
    data.M += np.diag(armature)
    return pin.computeGeneralizedGravity(model, data, q)


def compute_distance_between_shapes(
    rmodel: pin.Model, cmodel: pin.Model, shape1_id: int, shape2_id: int, q: np.ndarray
) -> float:
    """Computes the distance between shapes given the configuration vector.

    Args:
        rmodel (pin.Model): model of the robot
        cmodel (pin.Model): collision model of the robot
        shape1_id (int): id of the shape 1 in the collision model
        shape2_id (int): id of the shape 2 in the collision model
        q (np.ndarray): configuration array of the robot

    Returns:
        float: distance between the two given shapes
    """
    rdata = rmodel.createData()
    cdata = cmodel.createData()
    pin.forwardKinematics(rmodel, rdata, q)
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)
    req = hppfcl.DistanceRequest()
    res = hppfcl.DistanceResult()

    distance = hppfcl.distance(
        cmodel.geometryObjects[shape1_id].geometry,
        cdata.oMg[shape1_id],
        cmodel.geometryObjects[shape2_id].geometry,
        cdata.oMg[shape2_id],
        req,
        res,
    )

    return distance


def get_ee_pose_from_configuration(
    rmodel: pin.Model, rdata: pin.Data, id_ee_frame_id: int, q: np.ndarray
) -> pin.SE3:
    """Returns the SE3 describing the position of the end effector of the robot.

    Args:
        rmodel (pin.Model): Pinocchio Model of the robot.
        rdata (pin.Data): Pinocchio data of the robot
        id_ee_frame_name (int): ID of the frame of the end effector.
        q (np.ndarray): configuration of the robot.

    Returns:
        pin.SE3: SE3 of the position of the end effector of the robot.
    """
    pin.framesForwardKinematics(rmodel, rdata, q)
    pose = rdata.oMf[id_ee_frame_id]
    return pose.copy()


def get_last_joint(rmodel) -> Tuple[str, int, int]:
    """Returns the name of the last joint and it's id, with regards to the robot's type.

    Raises:
        UserWarning: Use of the UR instead of the panda for collision avoidance will not work.
        AttributeError: The pinocchio robot model given in the instanciation of the class has no last joint name
        similar to either the panda or the URs.

    Returns:
        Tuple[str, int, int]: Name of the last joint, ID of the last joint, ID of the last joint in the frames list.
    """

    if rmodel.existJointName("panda2_joint7"):
        last_joint_name = "panda2_joint7"
    elif rmodel.existJointName("panda_joint7"):
        last_joint_name = "panda_joint7"
    elif rmodel.existJointName("wrist_3_joint"):
        last_joint_name = "wrist_3_joint"
        warnings.warn(
            "Due to the no convexification of the geometry model, collisions will not be taken into consideration here."
        )
    else:
        raise AttributeError(
            "No known last joint name was found. Check the robot model. {}".format(
                str(rmodel)
            )
        )
    last_joint_id = rmodel.getJointId(last_joint_name)
    last_joint_frame_id = rmodel.getFrameId(last_joint_name)

    return (last_joint_name, last_joint_id, last_joint_frame_id)
