# default version in Ubuntu20.04
cmake_minimum_required(VERSION 3.16.3)

# Set up project properties
set(PROJECT_NAMESPACE agimus)
set(PROJECT_NAME agimus_controller)
set(PROJECT_DESCRIPTION
    "Defines the controller for the AGIMUS project."
)
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

add_project_dependency(
    pinocchio 2.6.21 REQUIRED PKG_CONFIG_REQUIRES "pinocchio >= 2.6.21")
add_project_dependency(
    crocoddyl 2.0.2 REQUIRED PKG_CONFIG_REQUIRES "crocoddyl >= 2.0.2")
add_project_dependency(
    example-robot-data 4.1.0 REQUIRED PKG_CONFIG_REQUIRES "example-robot-data >= 4.1.0")
add_project_dependency(
    mim_solvers 0.0.4 REQUIRED PKG_CONFIG_REQUIRES "mim_solvers >= 0.0.4")

if(NOT INSTALL_ROS_INTERFACE_ONLY)
    # Install the python package.
    file(GLOB project_python_source_files
         RELATIVE ${PROJECT_SOURCE_DIR}/${PROJECT_NAME} "${PROJECT_NAME}/*.py")
    foreach(file ${project_python_source_files})
        python_install_on_site(${PROJECT_NAME} ${file})
    endforeach()
    install(FILES package.xml DESTINATION share/${PROJECT_NAME})
endif()
install(DIRECTORY ${PROJECT_NAME}/resources DESTINATION ${PYTHON_SITELIB}/${PROJECT_NAME})


if(BUILD_ROS_INTERFACE)
    # Build with ros1 catkin specific configuration
    set(MY_CATKIN_DEPENDENCIES
        linear_feedback_controller_msgs
        rospy
    )
    find_package(catkin REQUIRED COMPONENTS ${MY_CATKIN_DEPENDENCIES})
    set(MY_PYTHON_DEPENDENCIES "")
    catkin_python_setup()
    catkin_package(
        CATKIN_DEPENDS ${MY_CATKIN_DEPENDENCIES}
        DEPENDS ${MY_PYTHON_DEPENDENCIES}
    )
endif()
