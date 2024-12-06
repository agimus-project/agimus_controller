# default version in Ubuntu20.04
cmake_minimum_required(VERSION 3.16.3)

# Set up project properties
set(PROJECT_NAMESPACE agimus)
set(PROJECT_NAME agimus_controller)
set(PROJECT_DESCRIPTION "Defines the controller for the AGIMUS project.")
set(PROJECT_URL https://github.com/${PROJECT_NAMESPACE}/${PROJECT_NAME})

# Create different building options
option(BUILD_ROS_INTERFACE "Build the ROS1 interface." ON)
option(INSTALL_ROS_INTERFACE_ONLY "Only build the ROS1 interface" OFF)

# Fetch the jrl-cmakemodules in order to build with it.
find_package(jrl-cmakemodules REQUIRED CONFIG)
get_property(
  JRL_CMAKE_MODULES
  TARGET jrl-cmakemodules::jrl-cmakemodules
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
include(${JRL_CMAKE_MODULES}/base.cmake)
include(${JRL_CMAKE_MODULES}/test.cmake)
include(${JRL_CMAKE_MODULES}/python.cmake)

# Declare the project from the above variables.
compute_project_args(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})

add_project_dependency(pinocchio 2.6.21 REQUIRED PKG_CONFIG_REQUIRES
                       "pinocchio >= 2.6.21")
add_project_dependency(crocoddyl 2.0.2 REQUIRED PKG_CONFIG_REQUIRES
                       "crocoddyl >= 2.0.2")
add_project_dependency(example-robot-data 4.0.9 REQUIRED PKG_CONFIG_REQUIRES
                       "example-robot-data >= 4.0.9")
add_project_dependency(mim_solvers 0.0.4 REQUIRED PKG_CONFIG_REQUIRES
                       "mim_solvers >= 0.0.4")

if(NOT INSTALL_ROS_INTERFACE_ONLY)
  # # cmake-format: off set(pkg1 ${PROJECT_NAME} hpp_interface.py mpc_search.py
  # mpc.py trajectory_buffer.py trajectory_point.py) set(pkg2
  # ${PROJECT_NAME}/hpp_panda planner.py scenes.py) set(pkg3
  # ${PROJECT_NAME}/main/panda main_hpp_mpc_buffer.py main_hpp_mpc.py
  # main_meshcat_display.py main_optim_traj.py main_scenes.py) set(pkg4
  # ${PROJECT_NAME}/main/ur3 main_hpp_mpc.py) set(pkg5 ${PROJECT_NAME}/ocps
  # ocp_croco_hpp.py ocp_pose_ref.py ocp.py) set(pkg6
  # ${PROJECT_NAME}/robot_model obstacle_params_parser.py panda_model.py
  # robot_model.py ur3_model.py) set(pkg7 ${PROJECT_NAME}/utils iostream.py
  # ocp_analyzer.py pin_utils.py process_handler.py) set(pkg8
  # ${PROJECT_NAME}/visualization plots.py wrapper_meshcat.py) set(packages pkg1
  # pkg2 pkg3 pkg4 pkg5 pkg6 pkg7 pkg8) # cmake-format: on
  foreach(package ${packages})
    list(POP_FRONT ${package} package_name)
    foreach(python_file ${${package}})
      python_install_on_site(${package_name} ${python_file})
    endforeach()
  endforeach()

  install(FILES package.xml DESTINATION share/${PROJECT_NAME})
endif()
install(DIRECTORY ${PROJECT_NAME}/resources
        DESTINATION ${PYTHON_SITELIB}/${PROJECT_NAME})

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(test)
endif()

if(BUILD_ROS_INTERFACE)
  # Build with ros1 catkin specific configuration
  set(MY_CATKIN_DEPENDENCIES linear_feedback_controller_msgs rospy)
  find_package(catkin REQUIRED COMPONENTS ${MY_CATKIN_DEPENDENCIES})
  set(MY_PYTHON_DEPENDENCIES "")
  catkin_python_setup()
  catkin_package(CATKIN_DEPENDS ${MY_CATKIN_DEPENDENCIES} DEPENDS
                 ${MY_PYTHON_DEPENDENCIES})
  catkin_install_python(
    PROGRAMS node/agimus_controller_node node/hpp_agimus_controller_node
    node/reaching_goal_controller_node DESTINATION
    ${CATKIN_PACKAGE_BIN_DESTINATION})
  install(PROGRAMS node/hpp_corbaserver
          DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}/node)
  install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
endif()
