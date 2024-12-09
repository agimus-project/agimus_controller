# agimus_controller

Whole-body model predictive controller to track a planned trajectory with ROS.

## Dependencies

- Humanoid Path Planner
- Agimus software
- Crocoddyl

## Installation

All of these dependencies are built in a single docker:
gitlab.laas.fr:4567/agimus-project/agimus_dev_container:noetic-devel

One can simply use this package in order to use the docker in the VSCode
development editor.
https://gitlab.laas.fr/agimus-project/agimus_dev_container

## Usage

### Simulation

#### Without ROS

Different mains are available:
- ur3 scripts :
    - `python3 -m agimus_controller.main.ur3.main_hpp_mpc -N=1`
- panda scripts :
    -  `python3 -m agimus_controller.main.panda.main_hpp_mpc_buffer`
    -  `python3 -m agimus_controller.main.panda.main_hpp_mpc`
    -  `python3 -m agimus_controller.main.panda.main_reaching_goal`
    -  `python3 -m agimus_controller.main.panda.main_meshcat_display`
    -  `python3 -m agimus_controller.main.panda.main_optim_traj`
    -  `python3 -m agimus_controller.main.panda.main_scenes`


#### With ROS

##### MPC nodes

Two mpc nodes can be launched :

mpc node with hpp trajectory given through topics :
```bash
roslaunch agimus_controller agimus_controller.launch
```
mpc node with hpp trajectory computed inside the node :
```bash
roslaunch agimus_controller hpp_agimus_controller.launch
```

##### Running simulation

the simulation can be launched with the following commands :
```bash
terminal 1 : roslaunch panda_torque_mpc simulation.launch arm_id:=panda simulate_camera:=false headless:=true
terminal 2 : roslaunch panda_torque_mpc sim_controllers.launch controller:=ctrl_mpc_linearized
terminal 3 : roslaunch agimus_controller agimus_controller.launch
terminal 4 : roslaunch agimus-hpp start_hpp.launch
terminal 5 : cd ~/ros_ws/src/agimus-demos/franka/manipulation/
python -i script_hpp.py
q_init, p = GrabAndDrop(robot, ps, binPicking, q_init, ros_bridge_config,vision_listener)
terminal 6 :
rosservice call /hpp/target/set_joint_names "names:
- 'pandas/panda_joint1'
- 'pandas/panda_joint2'
- 'pandas/panda_joint3'
- 'pandas/panda_joint4'
- 'pandas/panda_joint5'
- 'pandas/panda_joint6'
- 'pandas/panda_joint7'"
rostopic pub /hpp/target/read_path std_msgs/UInt32 0
rostopic pub /hpp/target/publish std_msgs/Empty
```

### Experiment

##### Environment variables setup

set ROS_IP and ROS_MASTER_URI :
```bash
export ROS_MASTER_URI=http://172.17.1.1:11311 ROS_IP=172.17.1.1 (laas)
```
set PANDA_IP :
```bash
export PANDA_IP=172.17.1.3 (laas)
```

##### Running experiment

The experiment can be launched with the following commands :

```bash
terminal 1 : roslaunch panda_torque_mpc real_controllers.launch controller:=ctrl_mpc_linearized robot:=panda robot_ip:=$PANDA_IP
terminal 2 : roslaunch agimus_controller agimus_controller.launch
terminal 3 : roslaunch agimus-hpp start_hpp.launch
terminal 4 : cd ~/ros_ws/src/agimus-demos/franka/manipulation/
python -i script_hpp.py
q_init, p = GrabAndDrop(robot, ps, binPicking, q_init, ros_bridge_config,vision_listener)
terminal 5 :
rosservice call /hpp/target/set_joint_names "names:
- 'pandas/panda_joint1'
- 'pandas/panda_joint2'
- 'pandas/panda_joint3'
- 'pandas/panda_joint4'
- 'pandas/panda_joint5'
- 'pandas/panda_joint6'
- 'pandas/panda_joint7'"
rostopic pub /hpp/target/read_path std_msgs/UInt32 0
rostopic pub /hpp/target/publish std_msgs/Empty
```
