import numpy as np
from agimus_controller.trajectory import (
    TrajectoryPoint,
    TrajectoryPointWeights,
    WeightedTrajectoryPoint,
)
from agimus_controller.ocp.ocp_croco_generic import BuildData
from agimus_controller.ocp import ocp_croco_generic
import unittest
import crocoddyl
import pinocchio


class OCPCrocoGenericBuilderTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self._rmodel = pinocchio.buildSampleModelManipulator()
        self._rgmodel = pinocchio.buildSampleGeometryModelManipulator(self._rmodel)
        self._state = crocoddyl.StateMultibody(self._rmodel)
        self._actuation = crocoddyl.ActuationModelFull(self._state)
        self._build_data = BuildData(self._state, self._actuation, self._rgmodel)

    def test_residual_model_state(self):
        xref = np.zeros(self._state.nx)
        differential = ocp_croco_generic.DifferentialActionModelFreeFwdDynamics(
            costs=[
                ocp_croco_generic.CostModelSumItem(
                    "state",
                    ocp_croco_generic.CostModelResidual(
                        residual=ocp_croco_generic.ResidualModelState(xref),
                        activation=ocp_croco_generic.ActivationModelWeightedQuad(1.0),
                    ),
                    update=True,
                )
            ]
        )
        cmodel = differential.build(self._build_data)
        cdata = cmodel.createData()
        x = np.random.random(self._state.nx)

        cmodel.calc(cdata, x)
        np.testing.assert_array_equal(cdata.costs.costs["state"].residual.r, x - xref)
        self.assertAlmostEqual(
            cdata.costs.costs["state"].cost, np.sum(0.5 * (x - xref) ** 2)
        )

        xref = np.random.random(self._state.nx)
        pt = WeightedTrajectoryPoint(
            point=TrajectoryPoint(
                robot_configuration=xref[: self._rmodel.nq],
                robot_velocity=xref[self._rmodel.nq :],
            ),
            weights=TrajectoryPointWeights(
                w_robot_configuration=0.5 * np.ones(self._rmodel.nv),
                w_robot_velocity=10 * np.ones(self._rmodel.nv),
            ),
        )
        w = pt.weights.w_robot_state

        differential.update(self._build_data, cmodel, pt)
        cmodel.calc(cdata, x)
        np.testing.assert_array_equal(cdata.costs.costs["state"].residual.r, x - xref)
        self.assertAlmostEqual(
            cdata.costs.costs["state"].cost, np.sum(0.5 * w * (x - xref) ** 2)
        )

    def test_residual_model_control(self):
        uref = np.zeros(self._actuation.nu)
        differential = ocp_croco_generic.DifferentialActionModelFreeFwdDynamics(
            costs=[
                ocp_croco_generic.CostModelSumItem(
                    "control",
                    ocp_croco_generic.CostModelResidual(
                        residual=ocp_croco_generic.ResidualModelControl(uref),
                        activation=ocp_croco_generic.ActivationModelWeightedQuad(1.0),
                    ),
                    update=True,
                )
            ]
        )
        cmodel = differential.build(self._build_data)
        cdata = cmodel.createData()
        x = np.random.random(self._state.nx)
        u = np.random.random(self._actuation.nu)

        cmodel.calc(cdata, x, u)
        np.testing.assert_array_equal(cdata.costs.costs["control"].residual.r, u - uref)
        self.assertAlmostEqual(
            cdata.costs.costs["control"].cost, np.sum(0.5 * (u - uref) ** 2)
        )

        uref = np.random.random(self._actuation.nu)
        pt = WeightedTrajectoryPoint(
            point=TrajectoryPoint(robot_effort=uref),
            weights=TrajectoryPointWeights(
                w_robot_effort=0.5 * np.ones(self._rmodel.nv)
            ),
        )
        w = pt.weights.w_robot_effort

        differential.update(self._build_data, cmodel, pt)
        cmodel.calc(cdata, x, u)
        np.testing.assert_array_equal(cdata.costs.costs["control"].residual.r, u - uref)
        self.assertAlmostEqual(
            cdata.costs.costs["control"].cost, np.sum(0.5 * w * (u - uref) ** 2)
        )


class OCPCrocoGenericTest(unittest.TestCase):
    def test_creation(self):
        # OCPCrocoGeneric()
        pass


if __name__ == "__main__":
    unittest.main()
