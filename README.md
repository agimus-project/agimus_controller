# croco_hpp_connection

## Getting started

In order to build this pacakge one need various complexe codes:

- Humanoid Path Planneur
- Agimus software
- Croccodyl

All of these are built in a single docker:
gitlab.laas.fr:4567/agimus-project/agimus_dev_container:noetic-devel

One can simply use this package in order to use the docker in the VSCode
development editor.
https://gitlab.laas.fr/agimus-project/agimus_dev_container

## Run the application.

Once the code is built one can run these in several terminals inside the docker:
- roscore
- hppcorbaserver
- gepetto-gui
- ipython3 -i -m agimus_controller.script
