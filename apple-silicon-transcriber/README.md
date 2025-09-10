# Apple Silicon Transcriber

A lightweight transcription server for Apple Silicon hosts. The service loads a
Whisper or sherpa-onnx model using `onnxruntime` with the
`CoreMLExecutionProvider` and exposes a simple ZeroMQ REP endpoint.

## Usage

```bash
python server.py --model /path/to/model.onnx --bind tcp://*:5557
```

Requests should be sent as a multipart message containing a JSON header and raw
PCM audio bytes. The server responds with the transcription text.

This project mirrors the architecture of the
[apple-silicon-detector](https://github.com/frigate-nvr/apple-silicon-detector)
used for object detection.
