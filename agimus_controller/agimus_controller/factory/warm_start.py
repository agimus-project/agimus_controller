from agimus_controller.warm_start_base import WarmStartBase


def _create_warm_start_from_previous_solution() -> WarmStartBase:
    pass


def _create_warm_start_from_diffusion_model() -> WarmStartBase:
    pass


def create_warm_start(name: str) -> WarmStartBase:
    if name == "from_previous_solution":
        return _create_warm_start_from_previous_solution()

    if name == "from_diffusion_model":
        return _create_warm_start_from_diffusion_model()
