import yaml
import example_robot_data
import pinocchio
import numpy as np
import crocoddyl

from agimus_controller.trajectory import (
    WeightedTrajectoryPoint,
    TrajectoryPoint,
    TrajectoryPointWeights,
)
from agimus_controller.ocp.ocp_croco_generic import (
    create_croco_dataclasses,
    as_dict,
    BuildData,
    IntegratedActionModelEuler,
    DifferentialActionModelFreeFwdDynamics,
    CostModelSumItem,
    CostModelResidual,
    ResidualModelControl,
)

assert __name__ == "__main__"

robot_model = example_robot_data.load("panda")
rmodel = pinocchio.buildReducedModel(
    robot_model.model,
    [
        robot_model.model.getJointId("panda_finger_joint1"),
        robot_model.model.getJointId("panda_finger_joint2"),
    ],
    pinocchio.neutral(robot_model.model),
)

state = crocoddyl.StateMultibody(rmodel)
actuation = crocoddyl.ActuationModelFull(state)
build_data = BuildData(state, actuation)

model_dict = yaml.safe_load(
    """
class: IntegratedActionModelEuler
differential:
    class: DifferentialActionModelFreeFwdDynamics
    costs:
    - name: control_reg
      update: true
      weight: 1.0
      cost:
          class: CostModelResidual
          activation:
              class: ActivationModelWeightedQuad
          residual:
              class: ResidualModelControl
    - name: state_reg
      update: true
      weight: 1.0
      cost:
          class: CostModelResidual
          activation:
              class: ActivationModelWeightedQuad
          residual:
              class: ResidualModelState
    - name: goal_tracking
      update: true
      weight: 1.0
      cost:
          class: CostModelResidual
          activation:
              class: ActivationModelWeightedQuad
          residual:
              class: ResidualModelFramePlacement
              id: 0
    """
)
model = create_croco_dataclasses(model_dict)
action_model = model.build(build_data)

wpt = WeightedTrajectoryPoint(
    TrajectoryPoint(
        robot_configuration=pinocchio.neutral(rmodel),
        robot_velocity=np.random.random(rmodel.nv),
        robot_effort=np.random.random(rmodel.nv),
        end_effector_poses={"panda_hand_tcp": pinocchio.SE3.Random()},
    ),
    TrajectoryPointWeights(
        w_robot_configuration=0.3 * np.ones(rmodel.nv),
        w_robot_velocity=0.2 * np.ones(rmodel.nv),
        w_robot_effort=0.1 * np.ones(rmodel.nv),
        w_end_effector_poses={"panda_hand_tcp": 10 * np.ones(6)},
    ),
)
model.update(build_data, action_model, wpt)

print(action_model)
print(action_model.differential.costs)

model = IntegratedActionModelEuler(
    DifferentialActionModelFreeFwdDynamics(
        costs=[
            CostModelSumItem(
                name="foo",
                cost=CostModelResidual(residual=ResidualModelControl()),
                weight=1.0,
            ),
        ]
    ),
    0.1,
)
d = as_dict(model)
print(yaml.safe_dump(d))
