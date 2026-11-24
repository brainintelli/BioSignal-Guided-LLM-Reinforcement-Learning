"""Strict request/reply transport for the exported SDK only."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import socket
import time


MARKER = b"\r\n###"


@dataclass(frozen=True)
class Endpoint:
    host: str = "127.0.0.1"
    port: int = 7215
    timeout: float = 60.0

    def __post_init__(self):
        if not self.host.strip() or not 1 <= self.port <= 65535:
            raise ValueError("Invalid MaxLab host/port")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be finite and greater than zero")


class Api:
    def __init__(self, connection, timeout):
        self.connection = connection
        self.timeout = timeout

    def send(self, message, timeout=None):
        budget = self.timeout if timeout is None else min(self.timeout, timeout)
        deadline = time.monotonic() + budget
        self.connection.settimeout(budget)
        self.connection.sendall(message.encode("latin1") + MARKER)
        data = bytearray()
        while MARKER not in data:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("MaxLab API response timed out")
            self.connection.settimeout(remaining)
            chunk = self.connection.recv(65536)
            if not chunk:
                raise ConnectionError("MaxLab API closed the connection before replying")
            data.extend(chunk)
            if len(data) > 8 * 1024 * 1024:
                raise RuntimeError("MaxLab API response exceeds 8 MiB")
        response = bytes(data).split(MARKER, 1)[0].decode("latin1")
        if response.strip().lower().startswith("error"):
            raise RuntimeError(f"MaxLab rejected {message.splitlines()[0][:100]}: {response[:200]}")
        return response


@contextmanager
def bind_sdk(mx, endpoint):
    """Bind every SDK call to one explicit endpoint and propagate API errors.

    The vendored SDK's default context swallows errors and overrides its port
    from the environment. This process runs one experiment at a time, so a
    scoped adapter is sufficient; no shared desktop SDK files are modified.
    """
    previous = mx.comm.api_context

    @contextmanager
    def api_context(*args, **kwargs):
        with socket.create_connection((endpoint.host, endpoint.port),
                                      timeout=min(5.0, endpoint.timeout)) as connection:
            yield Api(connection, endpoint.timeout)

    mx.comm.api_context = api_context
    try:
        yield
    finally:
        mx.comm.api_context = previous
