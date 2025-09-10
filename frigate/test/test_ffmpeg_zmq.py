import json
import logging
import threading
import time

import zmq

from frigate.ffmpeg.zmq import start_or_restart_ffmpeg, stop_ffmpeg
from frigate.log import LogPipe


def _mock_server(endpoint: str, frame: bytes) -> None:
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(endpoint)
    try:
        while True:
            msg = socket.recv()
            try:
                json.loads(msg.decode("utf-8"))
                socket.send(b"ok")
                continue
            except Exception:
                pass
            if msg == b"frame":
                socket.send(frame)
            elif msg == b"stop":
                socket.send(b"bye")
                break
            else:
                socket.send(b"")
    finally:
        socket.close()
        context.term()


def test_zmq_ffmpeg_client() -> None:
    endpoint = "tcp://127.0.0.1:5560"
    frame = b"abcd"
    t = threading.Thread(target=_mock_server, args=(endpoint, frame), daemon=True)
    t.start()
    time.sleep(0.1)
    logpipe = LogPipe("test")
    client = start_or_restart_ffmpeg(
        ["dummy"],
        logging.getLogger("test"),
        logpipe,
        frame_size=4,
        mode="zmq",
        zmq_endpoint=endpoint,
    )
    data = client.read(4)
    assert data == frame
    stop_ffmpeg(client, logging.getLogger("test"))
    logpipe.close()
