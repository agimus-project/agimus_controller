# Usage and General Docs

This section provides a short overview of how to use `agimus_controller` and links to the API reference.

## Goals

- Explain available OCP building blocks
- Show example YAML usage for `OCPCrocoGeneric`

## Costs overview

The main costs and residual types are implemented in `agimus_controller.ocp.ocp_croco_generic` and include:

- Activation models: `ActivationModelWeightedQuad`, `ActivationModelExp` (and legacy `ActivationModelQuadExp`)
- Residuals for state and control: `ResidualModelState`, `ResidualModelControl`, `ResidualModelControlGrav`
- Frame residuals: `ResidualModelFramePlacement`, `ResidualModelFrameTranslation`, `ResidualModelFrameRotation`, `ResidualModelFrameVelocity`
- Collision residuals: `ResidualDistanceCollision`, `ResidualDistanceCollision2`

See the API for details and constructor arguments: :doc:`../api/ocp_croco_generic`.

## Example (YAML snippet)

```yaml
running_model:
  differential:
    class: DifferentialActionModelFreeFwdDynamics
    costs:
      - name: state
        cost:
          class: CostModelResidual
          residual:
            class: ResidualModelState
          activation:
            class: ActivationModelWeightedQuad
            weights: 1.0

terminal_model:
  differential:
    class: DifferentialActionModelFreeFwdDynamics
    costs: []
```
