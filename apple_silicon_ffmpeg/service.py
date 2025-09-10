#!/usr/bin/env python3
"""Remote FFmpeg service for Apple Silicon.

This simple service exposes a ZMQ REP socket that accepts a multipart
message of [header_json, payload]. The header defines the ffmpeg command
arguments and the payload contains raw video or audio data which will be
fed to ffmpeg's stdin. The stdout of ffmpeg is returned as the reply.

The service always prepends `-hwaccel videotoolbox` to the ffmpeg command
to leverage Apple Silicon hardware acceleration.
"""
import argparse
import json
import subprocess as sp
from typing import List

import zmq


def run_ffmpeg(cmd: List[str], buffer: bytes) -> bytes:
    """Run ffmpeg with the provided command and input buffer."""
    full_cmd = ["ffmpeg", "-hwaccel", "videotoolbox"] + cmd
    process = sp.Popen(
        full_cmd,
        stdin=sp.PIPE,
        stdout=sp.PIPE,
        stderr=sp.PIPE,
    )
    out, _ = process.communicate(buffer)
    return out


def serve(endpoint: str) -> None:
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(endpoint)
    while True:
        frames = socket.recv_multipart()
        if not frames:
            socket.send(b"")
            continue
        header = json.loads(frames[0].decode("utf-8"))
        payload = frames[1] if len(frames) > 1 else b""
        cmd = header.get("cmd", [])
        try:
            result = run_ffmpeg(cmd, payload)
        except Exception as exc:  # noqa: BLE001
            result = str(exc).encode("utf-8")
        socket.send(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apple Silicon FFmpeg service")
    parser.add_argument("--bind", default="tcp://*:5555", help="ZMQ bind endpoint")
    args = parser.parse_args()
    serve(args.bind)


if __name__ == "__main__":
    main()
