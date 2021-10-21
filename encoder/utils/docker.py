import docker
import socket
from docker.errors import NotFound
from docker.models.containers import Container
from typing import Dict, Any
from contextlib import closing


def create_or_reuse_docker(
    image: str, startup_args: Dict[str, Any] = None, reuse_name: str = None
) -> (Container, bool):
    client = docker.from_env()
    startup_args = startup_args or {}
    is_reused = True
    try:
        container = client.containers.get(reuse_name)
    except NotFound:
        container = client.containers.create(image=image, **startup_args)
        is_reused = False
    container.start()
    return container, is_reused


def safe_exec_on_docker(container: Container, command: str):
    exit_code, _ = container.exec_run(cmd=command)
    if exit_code != 0:
        raise RuntimeError(
            f"Execution of command <{command}> "
            f"on container [{container.name}] "
            f"returned exit_code {exit_code}."
        )


def allocate_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
