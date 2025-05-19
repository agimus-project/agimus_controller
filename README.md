# agimus_controller

ROS package of a Model Predictive Controller (MPC) to track a planned trajectory.

## Dependencies

- Humanoid Path Planner
- Agimus software
- Crocoddyl

## Installation

All of these dependencies are built in a single docker:
gitlab.laas.fr:4567/agimus-project/agimus_dev_container

One can simply use this package in order to use the docker in the VSCode
development editor.
https://gitlab.laas.fr/agimus-project/agimus_dev_container

## Launch

There are 3 nodes available, an MPC, a trajectory publisher and a last one to view MPC predictions in rviz.
The two first nodes are sets by YAML files, you can see an example on how to launch them by looking at this demo [launch file](http://github.com/agimus-project/agimus-demos/blob/humble-devel/agimus_demo_03_mpc_dummy_traj/launch/bringup.launch.py).
The last node to view MPC predictions can be launched with :
```bash
ros2 run agimus_controller_ros mpc_debugger_node --frame frame --parent-frame parent-frame
```
### Parameters

Parameters from the YAML files are passed using the [generate_parameter_library](https://github.com/PickNikRobotics/generate_parameter_library).

#### agimus_controller_node

the agimus_controller_node is the MPC node, it is set by two YAML files :
- The first one sets the costs and constraints used inside the optimal control problem (OCP), you can find an example here [ocp_definition_file.yaml](https://github.com/agimus-project/agimus-demos/blob/humble-devel/agimus_demo_03_mpc_dummy_traj/config/ocp_definition_file.yaml).
- The second one sets all the other parameters for the solver, the building of the pinocchio models, and the node itself. Details about these parameters can be found [here](https://github.com/agimus-project/agimus_controller/blob/humble-devel/agimus_controller_ros/agimus_controller_ros/agimus_controller_parameters.yaml), also you can find an example here  [agimus_controller_params.yaml](https://github.com/agimus-project/agimus-demos/blob/humble-devel/agimus_demo_03_mpc_dummy_traj/config/agimus_controller_params.yaml).

#### simple_trajectory_publisher

This trajectory publisher node is set by one YAML file:
- Sets parameters for the trajectory and it's weights to send, details about these parameters can be found [here](https://github.com/agimus-project/agimus_controller/blob/humble-devel/agimus_controller_ros/agimus_controller_ros/trajectory_weights_parameters.yaml). An example can be found here [trajectory_weigths_params.yaml](https://github.com/agimus-project/agimus-demos/blob/humble-devel/agimus_demo_03_mpc_dummy_traj/config/trajectory_weigths_params.yaml).


## ROS API

### agimus_controller_node

#### Publishers

- **control** [linear_feedback_controller_msgs/msg/Control]

    Control output of the MPC.

- **mpc_debug** [agimus_msgs/msg/MpcDebug]

    Contains predictions of the solver, number of iterations, references and residuals of costs, KKT norm condition of the solver.

    Enabled by the parameter **publish_debug_data**.

- **ocp_solve_time** [builtin_interfaces/msg/Duration]

    Solve time of the solver used by the MPC.

    Enabled by the parameter **publish_debug_data**.

- **ocp_x0** [builtin_interfaces/msg/Duration]

    Current state of the robot used by the MPC, usefull to replay it offline.

    Enabled by the parameter **publish_debug_data**.

#### Subscribers

- **mpc_input** [agimus_msgs/msg/MpcInput]

    Contains references and weights of a trajectory point with the end-effector name.

- **sensor** [linear_feedback_controller_msgs/msg/Sensor]

    Current state of the robot.

- **robot_description** [std_msgs/msg/String]

    String containing URDF description of the robot.

- **environment_description** [std_msgs/msg/String]

    String containing URDF description of the robot's environment.

- **robot_srdf_description** [std_msgs/msg/String]

    String containing SRDF description of the robot.

- **joint_state** [sensor_msgs/msg/JointState]

    Give state of the robot before robot models are set.

### Simple_Trajectory_Publisher

#### Publishers

- **mpc_input** [agimus_msgs/msg/MpcInput]

    Contains references and weights of a trajectory point with the end-effector name.

#### Subscribers

- **sensor** [linear_feedback_controller_msgs/msg/Sensor]

    Current state of the robot.

- **robot_description** [std_msgs/msg/String]

    String containing URDF description of the robot.

### Mpc_debugger_node

#### Publishers


- **mpc_states_prediction_markers** [visualization_msgs/msg/MarkerArray]

    MPC states predictions markers arrays.

#### Subscribers

- **mpc_debug** [agimus_msgs/msg/MpcDebug]

    Contains predictions of the solver, number of iterations, references and residuals of costs, KKT norm condition of the solver.

    Enabled by the parameter **publish_debug_data**.

- **robot_description** [std_msgs/msg/String]

    String containing URDF description of the robot.

- **environment_description** [std_msgs/msg/String]

    String containing URDF description of the robot's environment.

- **robot_srdf_description** [std_msgs/msg/String]

    String containing SRDF description of the robot.

- **joint_state** [sensor_msgs/msg/JointState]

    Give state of the robot before robot models are set.
