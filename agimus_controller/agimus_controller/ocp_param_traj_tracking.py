from dataclasses import dataclass


from agimus_controller.ocp_param_base import OCPParamsBaseCroco


@dataclass
class OCPParamsTrajTracking(OCPParamsBaseCroco):
    """Input data structure of the OCP."""

    collision_safety_margin: float = 0.02  # safety margin for collision pairs
    activation_distance_threshold: float = 0.04  # activation thershold
