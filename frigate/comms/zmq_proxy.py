"""Facilitates communication over zmq proxy."""

import json
import os
import socket
import threading
from typing import Generic, TypeVar

import zmq
import logging

from frigate.const import FAST_QUEUE_TIMEOUT

SOCKET_PUB = "ipc:///tmp/cache/proxy_pub"
SOCKET_SUB = "ipc:///tmp/cache/proxy_sub"

HOST_SOCKET_PUB = os.getenv("FRIGATE_PROXY_HOST_PUB")
HOST_SOCKET_SUB = os.getenv("FRIGATE_PROXY_HOST_SUB")
# timeout (ms) used for host health checks and connects
HOST_CONNECT_TIMEOUT = int(os.getenv("FRIGATE_PROXY_TIMEOUT_MS", "1000"))


class ZmqProxyRunner(threading.Thread):
    def __init__(
        self,
        context: zmq.Context[zmq.Socket],
        host_pub: str | None,
        host_sub: str | None,
        timeout_ms: int,
    ) -> None:
        super().__init__(name="detection_proxy")
        self.context = context
        self.host_pub = host_pub
        self.host_sub = host_sub
        self.timeout_ms = timeout_ms

    def run(self) -> None:
        """Run the proxy."""
        incoming = self.context.socket(zmq.XSUB)
        incoming.bind(SOCKET_PUB)
        incoming.setsockopt(zmq.LINGER, 0)
        if self.host_pub:
            incoming.setsockopt(zmq.CONNECT_TIMEOUT, self.timeout_ms)
            try:
                incoming.connect(self.host_pub)
            except zmq.ZMQError:
                pass

        outgoing = self.context.socket(zmq.XPUB)
        outgoing.bind(SOCKET_SUB)
        outgoing.setsockopt(zmq.LINGER, 0)
        if self.host_sub:
            outgoing.setsockopt(zmq.CONNECT_TIMEOUT, self.timeout_ms)
            try:
                outgoing.connect(self.host_sub)
            except zmq.ZMQError:
                pass

        # Blocking: This will unblock (via exception) when we destroy the context
        # The incoming and outgoing sockets will be closed automatically
        # when the context is destroyed as well.
        try:
            zmq.proxy(incoming, outgoing)
        except zmq.ZMQError:
            pass


class ZmqProxy:
    """Proxies video and audio detections."""

    def __init__(self) -> None:
        host_pub = HOST_SOCKET_PUB
        host_sub = HOST_SOCKET_SUB
        timeout_ms = HOST_CONNECT_TIMEOUT

        # basic health check so a missing host service doesn't block startup
        for endpoint in [host_pub, host_sub]:
            if endpoint and endpoint.startswith("tcp://"):
                address, port = endpoint[6:].rsplit(":", 1)
                try:
                    with socket.create_connection(
                        (address, int(port)), timeout=timeout_ms / 1000
                    ):
                        pass
                except OSError:
                    # log a warning and disable host proxying for this endpoint
                    logging.warning(
                        "Host ZMQ endpoint %s unreachable; continuing without it", endpoint
                    )
                    if endpoint == host_pub:
                        host_pub = None
                    else:
                        host_sub = None

        self.context = zmq.Context()
        self.runner = ZmqProxyRunner(self.context, host_pub, host_sub, timeout_ms)
        self.runner.start()

    def stop(self) -> None:
        # destroying the context will tell the proxy to stop
        self.context.destroy()
        self.runner.join()


T = TypeVar("T")


class Publisher(Generic[T]):
    """Publishes messages."""

    topic_base: str = ""

    def __init__(self, topic: str = "") -> None:
        self.topic = f"{self.topic_base}{topic}"
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.connect(SOCKET_PUB)

    def publish(self, payload: T, sub_topic: str = "") -> None:
        """Publish message."""
        self.socket.send_string(f"{self.topic}{sub_topic} {json.dumps(payload)}")

    def stop(self) -> None:
        self.socket.close()
        self.context.destroy()


class Subscriber(Generic[T]):
    """Receives messages."""

    topic_base: str = ""

    def __init__(self, topic: str = "") -> None:
        self.topic = f"{self.topic_base}{topic}"
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)
        self.socket.connect(SOCKET_SUB)

    def check_for_update(self, timeout: float | None = FAST_QUEUE_TIMEOUT) -> T | None:
        """Returns message or None if no update."""
        try:
            has_update, _, _ = zmq.select([self.socket], [], [], timeout)

            if has_update:
                parts = self.socket.recv_string(flags=zmq.NOBLOCK).split(maxsplit=1)
                return self._return_object(parts[0], json.loads(parts[1]))
        except zmq.ZMQError:
            pass

        return self._return_object("", None)

    def stop(self) -> None:
        self.socket.close()
        self.context.destroy()

    def _return_object(self, topic: str, payload: T | None) -> T | None:
        return payload
