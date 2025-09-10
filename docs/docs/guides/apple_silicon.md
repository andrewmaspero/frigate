---
id: apple_silicon
title: Apple Silicon
---

# Apple Silicon

Frigate cannot access the Apple Neural Engine or VideoToolbox APIs from inside a container.  
This guide shows how to run companion services on the macOS host and expose them over ZMQ so the container can use Apple Silicon hardware.

## Host‑side services

### Object detection
1. Install the [Apple Silicon detector client](https://github.com/frigate-nvr/apple-silicon-detector).
2. Start the detector with a model and TCP endpoint:

```bash
make install
make run MODEL=/path/to/model.onnx ENDPOINT="tcp://*:5555"
```

### Audio transcription
Run a transcription server using `faster-whisper` or `mlx-whisper` and expose a ZMQ endpoint:

```bash
python3 -m frigate.tools.zmq_audio_transcriber --endpoint tcp://*:5556
```

### Embeddings
Start the embeddings service on another port:

```bash
python3 -m frigate.tools.zmq_embeddings --endpoint tcp://*:5557
```

### FFmpeg with VideoToolbox
FFmpeg inside the container cannot use VideoToolbox. Run it on the host and relay the stream:

```bash
ffmpeg -hide_banner -hwaccel videotoolbox -i rtsp://camera/stream \
  -c:v h264_videotoolbox -f flv tcp://*:8554/front
```

## Sample `config.yml`

```yaml
detectors:
  apple-silicon:
    type: zmq
    endpoint: tcp://host.docker.internal:5555

audio_transcription:
  enabled: True
  device: zmq
  endpoint: tcp://host.docker.internal:5556

semantic_search:
  enabled: True
  device: zmq
  endpoint: tcp://host.docker.internal:5557

cameras:
  front_door:
    ffmpeg:
      hwaccel_args: preset-videotoolbox-h264
      inputs:
        - path: tcp://host.docker.internal:8554/front
          roles:
            - detect
```

## Troubleshooting

- **Latency** – TCP adds overhead. Use `ipc://` endpoints when possible and keep services on the same host.
- **Timeouts** – Large models may take longer to respond. Increase `timeout` values in your config if requests fail.
- **Firewall** – macOS may block incoming ports. Allow each service under **System Settings → Network → Firewall**.
