"""ZMQ-based external object detection."""

import json
import logging
from typing import Any, Dict, List

import numpy as np

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig
from frigate.external_models.base import ModelTypeEnum, ZmqModelConfig, ZmqModelRunner

logger = logging.getLogger(__name__)


class ZmqObjectDetectionConfig(ZmqModelConfig):
    """Configuration for ZMQ object detection."""
    model_type: ModelTypeEnum = ModelTypeEnum.object_detection


class ZmqObjectDetector(ZmqModelRunner):
    """
    ZMQ-based external object detection using the new base framework.
    
    This provides the same functionality as the original ZmqIpcDetector
    but uses the unified ZmqModelRunner base class.
    """

    def __init__(self, config: ZmqObjectDetectionConfig, detector_config: BaseDetectorConfig):
        super().__init__(config)
        self.detector_config = detector_config
        # Preallocate zero result for error paths
        self._zero_result = np.zeros((20, 6), np.float32)

    def _build_header(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Build header for object detection request."""
        return {
            "model_type": self.config.model_type.value,
            "data_type": "tensor",
            "shape": list(input_data.shape),
            "dtype": str(input_data.dtype.name),
            "model_type_legacy": str(self.detector_config.model.model_type.name),
        }

    def _decode_response(self, frames: List[bytes]) -> np.ndarray:
        """Decode object detection response."""
        try:
            if len(frames) == 1:
                # Single-frame raw float32 (20x6)
                buf = frames[0]
                if len(buf) != 20 * 6 * 4:
                    logger.warning(
                        f"ZMQ object detector received unexpected payload size: {len(buf)}"
                    )
                    return self._zero_result
                return np.frombuffer(buf, dtype=np.float32).reshape((20, 6))

            if len(frames) >= 2:
                header = json.loads(frames[0].decode("utf-8"))
                shape = tuple(header.get("shape", [20, 6]))
                dtype = np.dtype(header.get("dtype", "float32"))
                return np.frombuffer(frames[1], dtype=dtype).reshape(shape)

            logger.warning("ZMQ object detector received empty reply")
            return self._zero_result
        except Exception as exc:
            logger.error(f"ZMQ object detector failed to decode response: {exc}")
            return self._zero_result

    def _get_fallback_result(self) -> np.ndarray:
        """Get fallback result on error."""
        return self._zero_result

    def detect_raw(self, tensor_input: np.ndarray) -> np.ndarray:
        """
        Raw detection interface compatible with DetectionApi.
        
        Args:
            tensor_input: Input tensor for object detection
            
        Returns:
            Detection array of shape (20, 6)
        """
        return self.process(tensor_input)