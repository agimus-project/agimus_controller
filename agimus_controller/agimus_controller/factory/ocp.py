from agimus_controller.ocp_base import OCPBase


def _create_ocp_hpp_crocco() -> OCPBase:
    pass


def _create_ocp_collision_avoidance() -> OCPBase:
    pass


def _create_ocp_single_ee_ref() -> OCPBase:
    pass


def create_ocp(name: str) -> OCPBase:
    if name == "hpp_crocco":
        return _create_ocp_hpp_crocco()

    if name == "collision_avoidance":
        return _create_ocp_collision_avoidance()

    if name == "single_ee_ref":
        return _create_ocp_single_ee_ref()
