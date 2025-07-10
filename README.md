# agimus_controller

Model Predictive Controller (MPC) to track a planned trajectory with ROS.

## Installation

The dependencies to built can be found in :
- [franka.repos](https://github.com/agimus-project/agimus-demos/blob/humble-devel/franka.repos)
- [control.repos](https://github.com/agimus-project/agimus-demos/blob/humble-devel/control.repos)
- [agimus_dev.repos](https://github.com/agimus-project/agimus-demos/blob/humble-devel/agimus_dev.repos)

## agimus_controller
This repository is in fact splitted in 3 packages working together, the first one is agimus_controller.

### Factory
 The factory contains classes to build the model of the robot with its environment easily, it also contains functions for creating ocp or warm start classes.

 ### OCP

 Few ocp are available, among them the [ocp_croco_generic](agimus_controller/agimus_controller/ocp/ocp_croco_generic.py) allows to create ocp easily by simply writing the costs/constraints inside a yaml file, an example for this can be found here [ocp_goal_reaching.yaml](agimus_controller/agimus_controller/ocp/ocp_goal_reaching.yaml)

 ### Plots

 Various plots for MPC are available, to visualize predictions of the solver, number of iterations, convergence of the solver etc ... an example of the usage can be found in this example [folder](agimus_controller_examples/agimus_controller_examples/main/panda_pick_and_place/)

 ### Trajectories

 Different trajectories builder are available :
 - sine_wave_configuration_space : trajectory of a sine wave in the configuration space of the robot.
 - sine_wave_cartesian_space : trajectory of a sine wave in the cartesian space.
 - sine_wave_cartesian_space_weight_increasing : trajectory of a sine wave in the cartesian space but the end-effector cost increase over time between the two extremas of the sine wave, depending on the tuning of the weights, it may starts to look like a target switching between the two extremas.
 - generic_trajectory : this trajectory awaits for the user to build it's own trajectory and add it. A function to convert arrays of q, dq, ddq into a list of trajectory points is made available.
- generic_trajectory_visual_servoing : just like the generic_trajectory it awaits for users trajectory. The user has to specify when he adds a trajectory from which index to which index visual servoing has to happen. A small state machine is used to manage the sending of the weights.
Increasing weights are used for the cost in position of the visual servoing, and also for collision avoidance. Parameters are available to decide what are the max values respectively for the translation part of the pose cost, the rotation part, and the collision avoidance also.
A small trick is happening to avoid taking too much time at stopping visual servoing, a parameter regarding the increasing weights concerns at what time we want to achieve a certain percentage of the maximum weight, the trick is to stop increasing the time at this moment.

## agimus_controller_examples

### Pick and place example
This packages contains some examples using agimus_controller package, it has an example of a pick and place with panda that can be launched with :
`python -im agimus_controller_examples.main.panda_pick_and_place.main`
it is possible to visualize the movement of the mpc in meshcat by doing in the terminal `app.display_path_meshcat(app.xs)` .

### Robot model sensibility
A script to evaluate model sensibilty of a robot is also available [here](agimus_controller_examples/agimus_controller_examples/main/model_sensibility/evaluate_model_sensibility.py)

### helpers
Also some helpers are available in the utils folder, to rebuild panda model or create MPC object offline.

## agimus_controller_ros
### Launch

There are 3 nodes available, an MPC, a trajectory publisher and a last one to view MPC predictions in rviz.
The two first nodes are sets by YAML files, you can see an example on how to launch them by looking at this demo [launch file](http://github.com/agimus-project/agimus-demos/blob/humble-devel/agimus_demo_03_mpc_dummy_traj/launch/bringup.launch.py).
The last node to view MPC predictions can be launched with :
```bash
ros2 run agimus_controller_ros mpc_debugger_node --frame frame --parent-frame parent-frame
```
#### Parameters

Parameters from the YAML files are passed using the [generate_parameter_library](https://github.com/PickNikRobotics/generate_parameter_library).

##### agimus_controller_node

the agimus_controller_node is the MPC node, it is set by two YAML files :
- The first one sets the costs and constraints used inside the optimal control problem (OCP), you can find an example here [ocp_definition_file.yaml](https://github.com/agimus-project/agimus-demos/blob/humble-devel/agimus_demo_03_mpc_dummy_traj/config/ocp_definition_file.yaml).
- The second one sets all the other parameters for the solver, the building of the pinocchio models, and the node itself. Details about these parameters can be found [here](https://github.com/agimus-project/agimus_controller/blob/humble-devel/agimus_controller_ros/agimus_controller_ros/agimus_controller_parameters.yaml), also you can find an example here  [agimus_controller_params.yaml](https://github.com/agimus-project/agimus-demos/blob/humble-devel/agimus_demo_03_mpc_dummy_traj/config/agimus_controller_params.yaml).

##### simple_trajectory_publisher

This trajectory publisher node is set by one YAML file:
- Sets parameters for the trajectory and it's weights to send, details about these parameters can be found [here](https://github.com/agimus-project/agimus_controller/blob/humble-devel/agimus_controller_ros/agimus_controller_ros/trajectory_weights_parameters.yaml). An example can be found here [trajectory_weigths_params.yaml](https://github.com/agimus-project/agimus-demos/blob/humble-devel/agimus_demo_03_mpc_dummy_traj/config/trajectory_weigths_params.yaml).


### ROS API

#### agimus_controller_node

##### Publishers

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

##### Subscribers

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

#### Simple_Trajectory_Publisher

##### Publishers

- **mpc_input** [agimus_msgs/msg/MpcInput]

    Contains references and weights of a trajectory point with the end-effector name.

##### Subscribers

- **sensor** [linear_feedback_controller_msgs/msg/Sensor]

    Current state of the robot.

- **robot_description** [std_msgs/msg/String]

    String containing URDF description of the robot.

#### Mpc_debugger_node

##### Publishers


- **mpc_states_prediction_markers** [visualization_msgs/msg/MarkerArray]

    MPC states predictions markers arrays.

##### Subscribers

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
