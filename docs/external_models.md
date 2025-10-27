# External Models for Frigate

This feature extends Frigate's ZMQ detector pattern to support running different types of models externally, allowing you to leverage specialized hardware like Apple Silicon neural cores for various AI tasks.

## Overview

The external models framework allows you to run the following model types outside the Frigate container:

- **Object Detection**: Same as the existing ZMQ detector
- **Audio Detection**: External audio classification and sound detection  
- **Text Embeddings**: External text embedding generation for semantic search
- **Vision Embeddings**: External image embedding generation for semantic search
- **Face Recognition**: External face recognition processing (future)

## Benefits

- **Performance**: Run models on specialized hardware (Apple Silicon, dedicated GPUs, etc.)
- **Flexibility**: Use any model framework (CoreML, ONNX, TensorFlow, PyTorch, etc.)
- **Scalability**: Distribute model processing across multiple machines
- **Resource Management**: Keep the Frigate container lightweight

## Configuration

### Object Detection (Enhanced ZMQ Detector)

```yaml
detectors:
  zmq:
    type: zmq
    endpoint: ipc:///tmp/cache/zmq_detector
    use_external_models: true  # Enable new framework
```

### Audio Detection

```yaml
# In camera configuration
cameras:
  front_door:
    audio:
      enabled: true
      # ... other audio settings
    external_models:
      audio:
        enabled: true
        endpoint: ipc:///tmp/cache/zmq_audio
```

### Embeddings

```yaml
semantic_search:
  enabled: true
  external_models:
    embeddings:
      text_enabled: true
      vision_enabled: true
      text_endpoint: ipc:///tmp/cache/zmq_text_embedding
      vision_endpoint: ipc:///tmp/cache/zmq_vision_embedding
      text_embedding_dim: 768
      vision_embedding_dim: 768
```

## External Model Server Protocol

All external models use a consistent ZMQ REQ/REP protocol:

### Request Format

```
Multipart message: [header_json_bytes, data_bytes]
```

Header JSON structure:
```json
{
  "model_type": "object_detection|audio_detection|text_embedding|vision_embedding",
  "data_type": "tensor|audio|text|image_batch",
  "shape": [height, width, channels],  // For tensor/audio data
  "dtype": "uint8|float32|int16",      // For tensor/audio data
  "embedding_dim": 768,                // For embedding models
  "batch_size": 1,                     // For batch processing
  "format": "jpg|png"                  // For image data
}
```

### Response Format

- **Object Detection**: Raw bytes of float32 array shape (20, 6) 
- **Audio Detection**: Raw bytes of float32 array shape (20, 6)
- **Embeddings**: Raw bytes of float32 embedding vectors

## Example Implementation

See `example_external_model_server.py` for a complete example of an external model server that handles all model types.

### Running the Example

1. Start the example server:
```bash
python example_external_model_server.py
```

2. Configure Frigate to use external models (see configuration above)

3. The example server will log all requests and return mock responses

## Apple Silicon Example

For Apple Silicon optimization, you could implement the external server using:

- **CoreML** for optimized inference
- **Metal Performance Shaders** for GPU acceleration
- **ONNX Runtime** with CoreML execution provider

```python
# Example CoreML integration
import coremltools as ct

class CoreMLModelServer(ExternalModelServer):
    def __init__(self):
        super().__init__()
        self.object_model = ct.models.MLModel('path/to/object_detection.mlmodel')
        self.audio_model = ct.models.MLModel('path/to/audio_detection.mlmodel')
        # ... etc
```

## Backward Compatibility

- Existing ZMQ detector continues to work unchanged
- Audio processing falls back to built-in models when external models are disabled
- Embeddings processing falls back to built-in ONNX models when external models are disabled
- Configuration is additive - no breaking changes

## Performance Considerations

- Use IPC endpoints for local processing (fastest)
- Use TCP endpoints for remote processing
- Batch processing is supported for embeddings
- Consider model warmup time on the external server
- Monitor ZMQ timeout settings for your hardware

## Troubleshooting

1. **Connection Issues**: Check that external server is running and endpoints match
2. **Timeout Errors**: Increase `request_timeout_ms` for slower hardware
3. **Data Format Errors**: Verify header JSON structure and data types
4. **Performance Issues**: Consider batching and model optimization

## Security Considerations

- IPC endpoints are only accessible from the local machine
- For TCP endpoints, consider using authentication and encryption
- Validate all input data in your external model server
- Run external servers with appropriate user permissions