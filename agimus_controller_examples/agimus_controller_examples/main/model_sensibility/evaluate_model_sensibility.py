import numpy as np
import matplotlib.pyplot as plt
import crocoddyl
import pinocchio as pin
import yaml
import example_robot_data


def get_next_state_delta_inertia(model, x, u, dt, link_idx, row_idx, col_idx, delta):
    """Integrate state to get next one, add small delta to link's inertia matrix."""
    curr_inertia = model.differential.pinocchio.inertias[link_idx].inertia.copy()
    new_inertia = curr_inertia.copy()
    new_inertia[row_idx, col_idx] += delta
    new_inertia[col_idx, row_idx] += delta
    model.differential.pinocchio.inertias[link_idx].inertia = new_inertia
    curr_dt = model.dt
    model.dt = dt
    d = model.createData()
    model.calc(d, x, u)
    model.dt = curr_dt
    model.differential.pinocchio.inertias[link_idx].inertia = curr_inertia
    return d.xnext.copy()


def get_next_state_delta_com(model, x, u, dt, link_idx, com_idx, delta):
    """Integrate state to get next one, add small delta to link's com pose."""
    curr_com = model.differential.pinocchio.inertias[link_idx].lever
    new_com = curr_com.copy()
    new_com[com_idx] += delta
    model.differential.pinocchio.inertias[link_idx].lever = new_com
    curr_dt = model.dt
    model.dt = dt
    d = model.createData()
    model.calc(d, x, u)
    model.dt = curr_dt
    model.differential.pinocchio.inertias[link_idx].lever = curr_com
    return d.xnext.copy()


def get_next_state_delta_mass(model, x, u, dt, link_idx, delta):
    """Integrate state to get next one, add small delta to link's mass."""
    model.differential.pinocchio.inertias[link_idx].mass += delta
    curr_dt = model.dt
    model.dt = dt
    d = model.createData()
    model.calc(d, x, u)
    model.dt = curr_dt
    model.differential.pinocchio.inertias[link_idx].mass -= delta
    return d.xnext.copy()


def get_reduced_panda_robot_model(robot):
    """Get pinocchio panda's reduced robot model."""
    q0 = np.zeros((9))
    locked_joints = [
        robot.model.getJointId("panda_finger_joint1"),
        robot.model.getJointId("panda_finger_joint2"),
    ]
    return pin.buildReducedModel(robot.model, locked_joints, np.array(q0))


if __name__ == "__main__":
    # retrieve x0 and u0 from real data on the robot
    with open("state_and_control_expe_data.yaml", "r") as file:
        state_control_expe_data = yaml.safe_load(file)

    # get model and set parameters
    robot = example_robot_data.load("panda")
    rmodel = get_reduced_panda_robot_model(robot)
    dt = 0.01  # integration step
    delta_inertia = 0.01
    delta_com = 0.01
    delta_mass = 0.01
    nq = rmodel.nq
    nv = rmodel.nv
    armature = np.array([0.1] * nq)
    point_idx = 2  # range from 1 to 5

    # get starting state and control
    point_data = state_control_expe_data[f"point_{point_idx}"]
    x0 = np.array(point_data["x0"])
    u0 = np.array(point_data["u0"])

    # create crocoddyl model
    state = crocoddyl.StateMultibody(rmodel)
    actuation = crocoddyl.ActuationModelFull(state)
    cost_model = crocoddyl.CostModelSum(state)
    differential_model = crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuation, cost_model
    )
    differential_model.armature = armature
    model = crocoddyl.IntegratedActionModelEuler(differential_model, dt)

    # compute sensibilities
    # for each link we have 10 values we're gonna test sensibility on,
    # 6 for inertia matrix, 3 for center of mass pose, 1 for mass
    model_sensibility = np.zeros(((nq + nv), nq * 10))
    x1_base = get_next_state_delta_inertia(model, x0, u0, dt, 0, 0, 0, 0)
    for link_idx in range(1, nq + 1):
        idx_link_inerta = (link_idx - 1) * 10
        for row_idx in range(3):
            for col_idx in range(0, row_idx + 1):
                x1 = get_next_state_delta_inertia(
                    model, x0, u0, dt, link_idx, row_idx, col_idx, delta_inertia
                )
                sensi_inertia = (x1 - x1_base) / delta_inertia
                model_sensibility[:, idx_link_inerta] = abs(sensi_inertia)
                idx_link_inerta += 1

    for link_idx in range(1, nq + 1):
        for com_idx in range(3):
            x1 = get_next_state_delta_com(
                model, x0, u0, dt, link_idx, com_idx, delta_com
            )
            sensi_com = (x1 - x1_base) / delta_com
            model_sensibility[:, 6 + (link_idx - 1) * 10 + com_idx] = abs(sensi_com)
        x1 = get_next_state_delta_mass(model, x0, u0, dt, link_idx, delta_mass)
        sensi_mass = (x1 - x1_base) / delta_mass
        model_sensibility[:, 9 + (link_idx - 1) * 10] = abs(sensi_mass)

    # plots
    u, s, vh = np.linalg.svd(model_sensibility)
    plt.plot(s, "+")
    plt.title("eigen values of sensibility matrix")
    plt.show()
    plt.imshow(np.abs(u))

    plt.colorbar()
    plt.title("eigen vectors on the left of sensibility matrix")
    plt.show()
    plt.imshow(np.abs(vh))
    plt.title("eigen vectors on the right of sensibility matrix")
    plt.colorbar()
    plt.show()
