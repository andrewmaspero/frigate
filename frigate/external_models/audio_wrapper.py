"""External audio detection wrapper for Frigate audio processing."""

import logging
from typing import List, Tuple

import numpy as np

from frigate.const import AUDIO_MIN_CONFIDENCE
from frigate.external_models.audio_detection import ZmqAudioDetectionConfig, ZmqAudioDetector
from frigate.object_detection.base import load_labels

logger = logging.getLogger(__name__)


class ExternalAudioDetector:
    """
    Wrapper for external ZMQ audio detection that's compatible with AudioTfl interface.
    
    This allows the existing audio processing code to optionally use external
    ZMQ-based audio detection instead of the built-in TensorFlow Lite model.
    """

    def __init__(self, zmq_config: ZmqAudioDetectionConfig, num_threads: int = 2):
        self.zmq_detector = ZmqAudioDetector(zmq_config)
        self.num_threads = num_threads
        self.labels = load_labels("/audio-labelmap.txt", prefill=521)

    def _detect_raw(self, tensor_input: np.ndarray) -> np.ndarray:
        """Raw detection using external ZMQ model."""
        return self.zmq_detector.detect_raw(tensor_input)

    def detect(self, tensor_input: np.ndarray, threshold: float = AUDIO_MIN_CONFIDENCE) -> List[Tuple[str, float, Tuple[float, float, float, float]]]:
        """
        Detect audio with the same interface as AudioTfl.detect().
        
        Args:
            tensor_input: Audio waveform as numpy array
            threshold: Minimum confidence threshold
            
        Returns:
            List of (label, score, bbox) tuples (bbox is dummy for audio)
        """
        detections = []

        raw_detections = self._detect_raw(tensor_input)

        for d in raw_detections:
            if d[1] < threshold:
                break
            detections.append(
                (self.labels[int(d[0])], float(d[1]), (d[2], d[3], d[4], d[5]))
            )
        return detections