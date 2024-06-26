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
include(${JRL_CMAKE_MODULES}/python.cmake)

# Declare the project from the above variables.
compute_project_args(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})

add_project_dependency(pinocchio 2.6.21 REQUIRED PKG_CONFIG_REQUIRES
                       "pinocchio >= 2.6.21")
add_project_dependency(crocoddyl 2.0.2 REQUIRED PKG_CONFIG_REQUIRES
                       "crocoddyl >= 2.0.2")
add_project_dependency(example-robot-data 4.1.0 REQUIRED PKG_CONFIG_REQUIRES
                       "example-robot-data >= 4.1.0")
add_project_dependency(mim_solvers 0.0.4 REQUIRED PKG_CONFIG_REQUIRES
                       "mim_solvers >= 0.0.4")

if(NOT INSTALL_ROS_INTERFACE_ONLY)
  # Install the python package.
  set(project_python_source_files hpp_interface.py mpc_search.py mpc.py
                                  trajectory_point.py)
  foreach(file ${project_python_source_files})
    python_install_on_site(${PROJECT_NAME} ${file})
  endforeach()

  # Install the panda planning files.
  set(project_python_hpp_panda_files planner.py scenes.py wrapper_panda.py)
  foreach(file ${project_python_hpp_panda_files})
    python_install_on_site(${PROJECT_NAME}/hpp_panda ${file})
  endforeach()

  # Install the main files.
  set(project_python_main_files main_hpp_mpc.py main_hpp_panda_mpc.py
                                main_mpc.py main_optim_traj.py)
  foreach(file ${project_python_main_files})
    python_install_on_site(${PROJECT_NAME}/main ${file})
  endforeach()

  # Install the ocp files.
  set(project_python_ocp_files ocp_croco_hpp.py ocp.py)
  foreach(file ${project_python_ocp_files})
    python_install_on_site(${PROJECT_NAME}/ocps ${file})
  endforeach()

  # Install the theos_work files.
  set(project_theos_work_python_source_files ocp_analyzer.py
                                             ocp_pick_and_place2_v1.py run.py)
  foreach(file ${project_theos_work_python_source_files})
    python_install_on_site(${PROJECT_NAME}/theos_work ${file})
  endforeach()

  # Install the utils files.
  set(project_python_utils_files
      build_models.py
      ocp_analyzer.py
      pin_utils.py
      plots.py
      ros_np_multiarray.py
      scenes.py
      wrapper_meshcat.py
      wrapper_panda.py)
  foreach(file ${project_python_utils_files})
    python_install_on_site(${PROJECT_NAME}/utils ${file})
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
    PROGRAMS
    node/agimus_controller_node
    node/hpp_agimus_controller_node
    node/gepetto_gui
    node/hpp_corbaserver
    DESTINATION
    ${CATKIN_PACKAGE_BIN_DESTINATION})
  install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
endif()
