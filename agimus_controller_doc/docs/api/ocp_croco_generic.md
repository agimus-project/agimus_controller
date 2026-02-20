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
ActivationModelWeightedQuad
ActivationModelExp
ActivationModelQuadExp
ResidualModelState
ResidualModelControl
ResidualModelControlGrav
ResidualModelFramePlacement
ResidualModelFrameTranslation
ResidualModelFrameRotation
ResidualModelFrameVelocity
ResidualDistanceCollision
ResidualDistanceCollision2
CostModelResidual
CostModelSumItem
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
