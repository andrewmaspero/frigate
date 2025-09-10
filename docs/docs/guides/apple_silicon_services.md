---
id: apple_silicon_services
title: Apple Silicon Host Services
---

# Apple Silicon Host Services

Frigate can offload resource intensive tasks to the macOS host in order to
use the Apple Neural Engine and VideoToolbox. Three companion services are
available:

- `apple-silicon-detector` – runs object detection on the Neural Engine.
- `apple-silicon-transcriber` – provides speech-to-text on the Neural Engine.
- `apple-silicon-ffmpeg` – handles video decoding/encoding with VideoToolbox.

## Prerequisites

1. macOS 12+ on Apple Silicon.
2. [Homebrew](https://brew.sh/) installed.

## Installation

### apple-silicon-detector

```bash
brew install frigate-cli/tap/apple-silicon-detector
brew services start apple-silicon-detector
```

The service listens on port **5555** by default.

### apple-silicon-transcriber

```bash
brew install frigate-cli/tap/apple-silicon-transcriber
brew services start apple-silicon-transcriber
```

The service listens on port **5556** by default.

### apple-silicon-ffmpeg

```bash
brew install frigate-cli/tap/apple-silicon-ffmpeg
brew services start apple-silicon-ffmpeg
```

The FFmpeg proxy listens on port **5558** and exposes VideoToolbox
acceleration.

## Configuration in Frigate

Use `device: zmq` and point to `host.docker.internal` to connect from the
container, for example:

```yaml
detectors:
  apple-silicon:
    type: onnx
    device: zmq
    endpoint: tcp://host.docker.internal:5555
```

```yaml
audio_transcription:
  enabled: False
  device: zmq
  endpoint: tcp://host.docker.internal:5556
```

```yaml
semantic_search:
  enabled: True
  device: zmq
  endpoint: tcp://host.docker.internal:5557
```

```yaml
ffmpeg:
  hwaccel_args: preset-videotoolbox-h264
  device: zmq
  endpoint: tcp://host.docker.internal:5558
```

Ensure each host service is running and that macOS allows incoming
connections for the binaries. Docker Desktop for Mac does not support
`--net=host`, so `host.docker.internal` must be used when connecting to
services on the host.
