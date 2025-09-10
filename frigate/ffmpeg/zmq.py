"""ZMQ-based remote FFmpeg client utilities."""
from __future__ import annotations

import logging
import subprocess as sp
from typing import Any, List

import zmq

from frigate.log import LogPipe


class ZmqRemoteFfmpegClient:
    def __init__(
        self,
        endpoint: str,
        cmd: List[str],
        request_timeout_ms: int = 1000,
        linger_ms: int = 0,
    ) -> None:
        self._context = zmq.Context()
        self._endpoint = endpoint
        self._request_timeout_ms = request_timeout_ms
        self._linger_ms = linger_ms
        self._socket = None
        self._create_socket()
        self._cmd = cmd
        self.pid = 0
        self._start()

    def _create_socket(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close(linger=self._linger_ms)
            except Exception:  # noqa: BLE001
                pass
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self._request_timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self._request_timeout_ms)
        self._socket.setsockopt(zmq.LINGER, self._linger_ms)
        self._socket.connect(self._endpoint)

    def _start(self) -> None:
        try:
            self._socket.send_json({"cmd": self._cmd})
            self._socket.recv()
        except Exception:  # noqa: BLE001
            pass

    def read(self, size: int) -> bytes:
        try:
            self._socket.send(b"frame")
            return self._socket.recv()
        except zmq.Again:
            try:
                self._create_socket()
            except Exception:  # noqa: BLE001
                pass
            return b""
        except zmq.ZMQError:
            try:
                self._create_socket()
            except Exception:  # noqa: BLE001
                pass
            return b""

    def terminate(self) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._socket.send(b"stop")
            self._socket.recv()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._socket.close(linger=self._linger_ms)
        except Exception:  # noqa: BLE001
            pass

    def poll(self) -> None:
        return None


def stop_ffmpeg(ffmpeg_process, logger: logging.Logger) -> None:
    if hasattr(ffmpeg_process, "close") and not isinstance(ffmpeg_process, sp.Popen):
        logger.info("Closing the existing ffmpeg client...")
        ffmpeg_process.close()
        return
    logger.info("Terminating the existing ffmpeg process...")
    ffmpeg_process.terminate()
    try:
        logger.info("Waiting for ffmpeg to exit gracefully...")
        ffmpeg_process.communicate(timeout=30)
    except sp.TimeoutExpired:
        logger.info("FFmpeg didn't exit. Force killing...")
        ffmpeg_process.kill()
        ffmpeg_process.communicate()


def start_or_restart_ffmpeg(
    ffmpeg_cmd: List[str],
    logger: logging.Logger,
    logpipe: LogPipe,
    frame_size: int | None = None,
    ffmpeg_process: Any = None,
    mode: str = "local",
    zmq_endpoint: str | None = None,
):
    if mode == "zmq" and zmq_endpoint:
        if ffmpeg_process is not None and hasattr(ffmpeg_process, "close"):
            ffmpeg_process.close()
        return ZmqRemoteFfmpegClient(zmq_endpoint, ffmpeg_cmd)

    if ffmpeg_process is not None:
        stop_ffmpeg(ffmpeg_process, logger)

    if frame_size is None:
        process = sp.Popen(
            ffmpeg_cmd,
            stdout=sp.DEVNULL,
            stderr=logpipe,
            stdin=sp.DEVNULL,
            start_new_session=True,
        )
    else:
        process = sp.Popen(
            ffmpeg_cmd,
            stdout=sp.PIPE,
            stderr=logpipe,
            stdin=sp.DEVNULL,
            bufsize=frame_size * 10,
            start_new_session=True,
        )
    return process
