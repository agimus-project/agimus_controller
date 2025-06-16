from agimus_controller_examples.utils.process_handler import (
    HppCorbaServer,
    GepettoGuiServer,
    MeshcatServer,
)


class Servers(object):
    def __init__(self) -> None:
        self._servers = []

    def is_running(self):
        is_running = True
        for server in self._servers:
            is_running = is_running and server.is_running()
        return is_running

    def start(self):
        for server in self._servers:
            server.start()

    def stop(self):
        for server in self._servers:
            server.stop()

    def spawn_servers(self, use_gui):
        self._servers.append(HppCorbaServer())
        if use_gui:
            self._servers.append(GepettoGuiServer())

    def spawn_servers_meshcat(self, use_gui):
        self._servers.append(HppCorbaServer())
        if use_gui:
            self._servers.append(MeshcatServer())
