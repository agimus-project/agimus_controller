import numpy as np
import matplotlib.pyplot as plt


def return_state_vector(ddp):
    """Creates an array containing the states along the horizon

    Arguments:
        dpp -- Crocoddyl object containing all solver related data
    """
    raw_states = np.array(ddp.xs)

    nq = int((len(raw_states[0]) - 7) / 2 + 3)

    states = dict(
        base_position=raw_states[:, 0:3],
        base_linear_velocity=raw_states[:, nq : (nq + 3)],
        base_orientation=raw_states[:, 3:7],
        base_angular_velocity=raw_states[:, (nq + 3) : (nq + 6)],
        joints_position=raw_states[:, 7:nq],
        joints_velocity=raw_states[:, (nq + 6) :],
    )

    return states


def return_command_vector(ddp):
    """Creates an array containing the commands along the horizon

    Arguments:
        dpp -- Crocoddyl object containing all solver related data
    """
    commands = np.array(ddp.us)

    return commands


def return_cost_vectors(ddp, weighted=False, integrated=False):
    """
    Creates a dictionary with the costs along the horizon from the ddp object and returns it
    """
    costs = {}
    for i in range(ddp.problem.T):
        for cost_tag in list(
            ddp.problem.runningModels[i].differential.costs.costs.todict().keys()
        ):
            if cost_tag not in costs:
                costs.update({cost_tag: np.nan * np.ones(ddp.problem.T + 1)})
            try:
                costs[cost_tag][i] = (
                    ddp.problem.runningDatas[i].differential.costs.costs[cost_tag].cost
                )
            except Exception as ex:
                print(ex.with_traceback())
                costs[cost_tag][i] = np.mean(
                    [
                        diff.costs.costs[cost_tag].cost
                        for diff in ddp.problem.runningDatas[i].differential
                    ]
                )
            if weighted:
                costs[cost_tag][i] *= (
                    ddp.problem.runningModels[i]
                    .differential.costs.costs.todict()[cost_tag]
                    .weight
                )
                if i == ddp.problem.T - 1:
                    print("i ", i)
                    print("cost tag ", cost_tag)
                    print(
                        "weight",
                        ddp.problem.runningModels[i]
                        .differential.costs.costs.todict()[cost_tag]
                        .weight,
                    )
                    print("cost ")

            if integrated:
                costs[cost_tag][i] *= ddp.problem.runningModels[i].dt

    for cost_tag in list(
        ddp.problem.terminalModel.differential.costs.costs.todict().keys()
    ):
        if cost_tag not in costs:
            costs.update({cost_tag: np.nan * np.ones(ddp.problem.T + 1)})
        try:
            costs[cost_tag][-1] = ddp.problem.terminalData.differential.costs.costs[
                cost_tag
            ].cost
            if weighted:
                costs[cost_tag][
                    -1
                ] *= ddp.problem.terminalModel.differential.costs.costs.todict()[
                    cost_tag
                ].weight
        except Exception as ex:
            print(ex.with_traceback())
            costs[cost_tag][-1] = np.mean(
                [
                    diff.costs.costs[cost_tag].cost
                    for diff in ddp.problem.terminalData.differential
                ]
            )
        # if weighted:
        #     costs[cost_tag][-1] *= ddp.problem.terminalModel.differential.costs.costs.todict()[cost_tag].weight

    return costs


def return_constraint_vector(solver):
    """
    Returns a dictionary with constraints along the horizon from the solver object.
    """
    constraints = {}
    for i in range(solver.problem.T):
        for constraint_tag, constraint_value in (
            solver.problem.runningDatas[i]
            .differential.constraints.constraints.todict()
            .items()
        ):
            if constraint_tag not in constraints:
                # Initialiser l'entrée du dictionnaire avec des NaN, en considérant la dimension du vecteur
                if (
                    isinstance(constraint_value.residual.r, np.ndarray)
                    and len(constraint_value.residual.r) > 1
                ):
                    num_components = len(constraint_value.residual.r)
                else:
                    num_components = 1
                constraints[constraint_tag] = np.nan * np.ones(
                    (solver.problem.T + 1, num_components)
                )
            try:
                if num_components == 1:
                    constraints[constraint_tag][i] = constraint_value.residual.r
                else:
                    constraints[constraint_tag][i, :] = constraint_value.residual.r
            except Exception as ex:
                print(ex.with_traceback())
                print("Error processing constraint:", constraint_tag)
    return constraints


def return_weights(ddp):
    weights = {}
    for i in range(ddp.problem.T):
        for cost_tag in list(
            ddp.problem.runningModels[i].differential.costs.costs.todict().keys()
        ):
            if cost_tag not in weights:
                weights.update({cost_tag: np.nan * np.ones(ddp.problem.T + 1)})
            weights[cost_tag][i] = (
                ddp.problem.runningModels[i]
                .differential.costs.costs.todict()[cost_tag]
                .weight
            )

    for cost_tag in list(
        ddp.problem.terminalModel.differential.costs.costs.todict().keys()
    ):
        if cost_tag not in weights:
            weights.update({cost_tag: np.nan * np.ones(ddp.problem.T + 1)})
        weights[cost_tag][
            -1
        ] = ddp.problem.terminalModel.differential.costs.costs.todict()[cost_tag].weight

    return weights


def return_time_vector(ddp, t0=0):
    """
    Returns a vector with the time evolution related to a ddp problem,
    useful when plotting and dt changes from node to node
    """
    time = np.zeros(ddp.problem.T + 1)
    if t0 != 0:
        time[0] = t0
    for i in range(1, ddp.problem.T + 1):
        time[i] = time[i - 1] + ddp.problem.runningModels[i - 1].dt
    return time


def plot_costs_from_dic(dic):
    """
    Plots dictionary tags
    """
    fig, ax_running = plt.subplots()
    ax_terminal = ax_running.twinx()
    runningCosts = []
    terminalCosts = []
    if "time" in dic:
        time = dic["time"]
    else:
        time = np.arange(0, len(dic[list(dic.keys())[0]]))
    for tag in list(dic.keys()):
        if np.sum(np.isnan(dic[tag])) == 0:
            ax_running.plot(dic[tag])
            runningCosts.append(tag)
        else:
            ax_terminal.scatter(time, dic[tag])
            terminalCosts.append(tag)
    ax_running.legend(runningCosts)
    ax_terminal.legend(terminalCosts)
    plt.show()


def plot_constraints_from_dic(dic, subplots_per_fig=6, nan_threshold=0.5, point_size=8):
    """
    Plots the constraints from the dictionary. If the data is sparse (i.e., contains many NaNs),
    it uses scatter plots with large points instead of line plots.

    Args:
    dic (dict): Dictionary containing the constraints.
    subplots_per_fig (int): Number of subplots per figure.
    nan_threshold (float): Proportion threshold of NaNs to switch from line plot to scatter plot.
    point_size (int): Size of the points in the scatter plot.
    """
    plot_count = 0
    keys = list(dic.keys())

    for key in keys:
        data = dic[key]

        if isinstance(data, np.ndarray) and data.ndim > 1:
            num_components = data.shape[1]
        else:
            num_components = 1

        total_subplots = num_components

        for subplot_idx in range(total_subplots):
            if plot_count % subplots_per_fig == 0:
                if plot_count > 0:
                    plt.show()
                fig = plt.figure()

            ax = fig.add_subplot(
                int(np.ceil(subplots_per_fig / 2)),
                2,
                (plot_count % subplots_per_fig) + 1,
            )

            if num_components == 1:
                component_data = data
            else:
                component_data = data[:, subplot_idx]

            # Calculer la densité des valeurs non-NaN
            non_nan_count = np.count_nonzero(~np.isnan(component_data))
            total_count = len(component_data)
            density = non_nan_count / total_count

            if density < nan_threshold:
                # Utiliser un scatter plot si la densité est faible
                ax.scatter(range(total_count), component_data, s=point_size)
            else:
                # Utiliser un line plot sinon
                ax.plot(component_data)

            if num_components == 1:
                ax.set_title(f"{key}")
            else:
                ax.set_title(f"{key} - Composant {subplot_idx + 1}")

            plot_count += 1

    if plot_count % subplots_per_fig != 0:
        plt.show()


def plot_state_from_dic(dic):
    fig = plt.figure()
    for i in range(len(list(dic.keys()))):
        key = list(dic.keys())[i]
        ax = fig.add_subplot(3, 2, i + 1)
        ax.plot(dic[key])
        ax.set_title(key)


def plot_command(commands):
    plt.figure()
    plt.plot(commands)
    plt.title("Commands")
