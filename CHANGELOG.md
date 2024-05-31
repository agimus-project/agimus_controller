# Changelog for package agimus_controller

## [Unreleased]

### Added:

In this first implementation we are in the creation phase of the package.
We included the work of TheoMF and create a packaging around it.
His work aims at creating a whole body model predictive controller that tracks
the humanoid-path-planner (HPP) plans.
First changes are:
- the packaging
- renaming some python files for more clarity

Commits:
- add precommit
- Fix typo and add hpp-manipulation-corba
- Add the list of python files by hand
- Merge pull request `#11 <https://github.com/agimus-project/agimus_controller/issues/11>`_ from agimus-project/topic/tmartinez/mpc-debug
  put hpp planification outside of the problem class
- Merge pull request `#10 <https://github.com/agimus-project/agimus_controller/issues/10>`_ from TheoMF/topic/tmartinez/mpc-debug
  put hpp planification outside of the problem class
- improve readability, change file name
- stop using Subpath class, put hpp's plan in one np array
- move hpp plan outside problem class, start implementing trajectory buffer
- add hpp acceleration plot, little fix on reset ocp function
- add debug code
- Merge pull request `#9 <https://github.com/agimus-project/agimus_controller/issues/9>`_ from TheoMF/noetic-devel
  Noetic devel
- clean cmakelists.txt
- Update README
- fix packaging and run of the agimus_controller
- remove the CMake poolicy setter
- Add packaging ROS1 and standard
- Move the code files in agimus_controller folder
- Move the code files in src
- Merge pull request `#7 <https://github.com/agimus-project/agimus_controller/issues/7>`_ from TheoMF/working
  add mpc code, clean code
- add debuging code
- compute hpp acceleration, change lists by np array
- add results in dictionary, add max increase in control information
- minor fixes to adapt to different robots
- add mpc code, clean code
- add hpp script, change dt
- fix horizon computation, change ocp formulation
- add function to plot integrated configuration
- few changes
- add more constraints for mim_solvers
- allow to run without hpp dependency
- allow to use mim_solvers, refactor code
- add class to search best costs, plot trajectories and display on viewer
- add class to create crocoddyl problem and solve it
- Initial commit
- Contributors: Maximilien Naveau, Naveau, TheoMF, Théo MARTINEZ, Théo Martinez

### Changed

### Removed