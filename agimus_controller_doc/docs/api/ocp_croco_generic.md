---
title: OCP Crocoddyl Generic API
---

# AGIMUS controller: generic croccodyl OCP


## Costs — quick index

Below are the main cost and residual classes implemented in this module. Click the class names to open full API docs.

### Activation Models

- [ActivationModelWeightedQuad](ActivationModelWeightedQuad.md) — Weighted quadratic activation
- [ActivationModelExp](ActivationModelExp.md) — Exponential activation
- [ActivationModelQuadExp](ActivationModelQuadExp.md) — Quadratic-exponential activation

### Residual Models

- [ResidualModelState](ResidualModelState.md) — State tracking residual
- [ResidualModelControl](ResidualModelControl.md) — Control regularization
- [ResidualModelControlGrav](ResidualModelControlGrav.md) — Gravity-compensated control
- [ResidualModelFramePlacement](ResidualModelFramePlacement.md) — 6D frame placement
- [ResidualModelFrameTranslation](ResidualModelFrameTranslation.md) — 3D frame translation
- [ResidualModelFrameRotation](ResidualModelFrameRotation.md) — 3D frame rotation
- [ResidualModelFrameVelocity](ResidualModelFrameVelocity.md) — Frame velocity tracking
- [ResidualDistanceCollision](ResidualDistanceCollision.md) — Collision distance (basic)
- [ResidualDistanceCollision2](ResidualDistanceCollision2.md) — Collision distance (advanced)

### Cost Models

- [CostModelResidual](CostModelResidual.md) — Residual-based cost
- [CostModelSumItem](CostModelSumItem.md) — Weighted cost sum item

## Example usage

```python
from agimus_controller.ocp.ocp_croco_generic import OCPCrocoGeneric, ResidualModelFramePlacement

# Load an OCP from YAML and inspect available cost names
ocp = OCPCrocoGeneric(...)
print(ocp._data.running_model.differential.costs)

# Programmatically create a Placement residual
# residual = ResidualModelFramePlacement(id="tool0")
```

## Module API

This page exposes the API of the `ocp_croco_generic` module.

```{automodule} agimus_controller.ocp.ocp_croco_generic
:members:
:undoc-members:
:show-inheritance:
```
