---
title: OCP Crocoddyl Generic API
---

# `agimus_controller.ocp.ocp_croco_generic`

This page exposes the API of the `ocp_croco_generic` module.

```{automodule} agimus_controller.ocp.ocp_croco_generic
:members:
:undoc-members:
:show-inheritance:
```

## Costs — quick index

Below are the main cost and residual classes implemented in this module. Click the class names above to open full API docs.

```{autosummary}
agimus_controller.ocp.ocp_croco_generic.ActivationModelWeightedQuad
agimus_controller.ocp.ocp_croco_generic.ActivationModelExp
agimus_controller.ocp.ocp_croco_generic.ActivationModelQuadExp
agimus_controller.ocp.ocp_croco_generic.ResidualModelState
agimus_controller.ocp.ocp_croco_generic.ResidualModelControl
agimus_controller.ocp.ocp_croco_generic.ResidualModelControlGrav
agimus_controller.ocp.ocp_croco_generic.ResidualModelFramePlacement
agimus_controller.ocp.ocp_croco_generic.ResidualModelFrameTranslation
agimus_controller.ocp.ocp_croco_generic.ResidualModelFrameRotation
agimus_controller.ocp.ocp_croco_generic.ResidualModelFrameVelocity
agimus_controller.ocp.ocp_croco_generic.ResidualDistanceCollision
agimus_controller.ocp.ocp_croco_generic.ResidualDistanceCollision2
agimus_controller.ocp.ocp_croco_generic.CostModelResidual
agimus_controller.ocp.ocp_croco_generic.CostModelSumItem
```

## Example usage

```python
from agimus_controller.ocp.ocp_croco_generic import OCPCrocoGeneric, ResidualModelFramePlacement

# Load an OCP from YAML and inspect available cost names
ocp = OCPCrocoGeneric(...)
print(ocp._data.running_model.differential.costs)

# Programmatically create a Placement residual
# residual = ResidualModelFramePlacement(id="tool0")
```
