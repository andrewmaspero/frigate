"""ZMQ-based external audio detection."""

import json
import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from frigate.external_models.base import ModelTypeEnum, ZmqModelConfig, ZmqModelRunner

logger = logging.getLogger(__name__)


class ZmqAudioDetectionConfig(ZmqModelConfig):
    """Configuration for ZMQ audio detection."""
    model_type: ModelTypeEnum = ModelTypeEnum.audio_detection


class ZmqAudioDetector(ZmqModelRunner):
    """
    ZMQ-based external audio detection.
    
    Compatible with the existing AudioTfl interface but processes audio
    through an external ZMQ endpoint instead of a local TensorFlow Lite model.
    """

    def __init__(self, config: ZmqAudioDetectionConfig):
        super().__init__(config)
        # Preallocate zero result for error paths (same format as AudioTfl)
        self._zero_result = np.zeros((20, 6), np.float32)

    def _build_header(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Build header for audio detection request."""
        return {
            "model_type": self.config.model_type.value,
            "data_type": "audio",
            "shape": list(input_data.shape),
            "dtype": str(input_data.dtype.name),
            "sample_rate": 16000,  # Standard audio sample rate for Frigate
        }

    def _decode_response(self, frames: List[bytes]) -> np.ndarray:
        """Decode audio detection response."""
        try:
            if len(frames) == 1:
                # Single-frame raw float32 (20x6) - same format as object detection
                buf = frames[0]
                if len(buf) != 20 * 6 * 4:
                    logger.warning(
                        f"ZMQ audio detector received unexpected payload size: {len(buf)}"
                    )
                    return self._zero_result
                return np.frombuffer(buf, dtype=np.float32).reshape((20, 6))

            if len(frames) >= 2:
                header = json.loads(frames[0].decode("utf-8"))
                shape = tuple(header.get("shape", [20, 6]))
                dtype = np.dtype(header.get("dtype", "float32"))
                return np.frombuffer(frames[1], dtype=dtype).reshape(shape)

            logger.warning("ZMQ audio detector received empty reply")
            return self._zero_result
        except Exception as exc:
            logger.error(f"ZMQ audio detector failed to decode response: {exc}")
            return self._zero_result

    def _get_fallback_result(self) -> np.ndarray:
        """Get fallback result on error."""
        return self._zero_result

    def detect(self, tensor_input: np.ndarray, threshold: float = 0.5) -> List[Tuple[str, float, Tuple[float, float, float, float]]]:
        """
        Detect audio with the same interface as AudioTfl.detect().
        
        Args:
            tensor_input: Audio waveform as numpy array
            threshold: Minimum confidence threshold
            
        Returns:
            List of (label, score, bbox) tuples (bbox is dummy for audio)
        """
        # Process through external model
        raw_detections = self.process(tensor_input)
        
        detections = []
        for d in raw_detections:
            if d[1] < threshold:
                break
            # Note: For audio, we'd need label mapping. For now, using index as placeholder
            # In real implementation, this would use the same label mapping as AudioTfl
            label = f"audio_class_{int(d[0])}"  # Placeholder - would need actual labelmap
            detections.append(
                (label, float(d[1]), (d[2], d[3], d[4], d[5]))
            )
        return detections

    def detect_raw(self, tensor_input: np.ndarray) -> np.ndarray:
        """
        Raw detection interface compatible with ZmqIpcDetector.
        
        Args:
            tensor_input: Audio waveform as numpy array
            
        Returns:
            Detection array of shape (20, 6)
        """
        return self.process(tensor_input)