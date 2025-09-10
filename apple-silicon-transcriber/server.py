import argparse
import json
from typing import Any

import numpy as np
import onnxruntime as ort
import zmq


# Simple ZMQ based transcription service for Apple Silicon machines.
#
# The server loads an ONNX model (Whisper or sherpa-onnx) using the
# CoreMLExecutionProvider to leverage the Apple Neural Engine. Requests are
# handled over a ZMQ REP socket. Each request is a multipart message containing
# a JSON header followed by raw PCM bytes. The server returns the transcription
# text as a UTF-8 string.


def load_session(model_path: str) -> ort.InferenceSession:
    """Load an ONNX model with CoreML and CPU providers."""
    return ort.InferenceSession(
        model_path,
        providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Apple Silicon transcription server")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument(
        "--bind",
        default="tcp://*:5557",
        help="ZMQ bind endpoint, e.g. tcp://*:5557",
    )
    args = parser.parse_args()

    session = load_session(args.model)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(args.bind)

    input_name = session.get_inputs()[0].name

    while True:
        header_bytes, audio_bytes = socket.recv_multipart()
        header = json.loads(header_bytes.decode("utf-8"))
        dtype = np.dtype(header.get("dtype", "float32"))
        shape = header.get("shape", (-1,))
        audio = np.frombuffer(audio_bytes, dtype=dtype).reshape(shape)

        outputs = session.run(None, {input_name: audio})
        # Assume first output represents transcription tokens or string
        text: Any = outputs[0]
        if isinstance(text, bytes):
            result = text.decode("utf-8")
        else:
            result = str(text)

        socket.send_string(result)


if __name__ == "__main__":
    main()
