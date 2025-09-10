"""Base class for external model processing via ZMQ."""

import json
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import zmq
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModelTypeEnum(str, Enum):
    """Supported external model types."""
    object_detection = "object_detection"
    audio_detection = "audio_detection"
    text_embedding = "text_embedding"
    vision_embedding = "vision_embedding"
    face_recognition = "face_recognition"


class ZmqModelConfig(BaseModel):
    """Base configuration for ZMQ external models."""
    endpoint: str = Field(title="ZMQ IPC endpoint")
    model_type: ModelTypeEnum = Field(title="Type of model")
    request_timeout_ms: int = Field(default=200, title="ZMQ request timeout in milliseconds")
    linger_ms: int = Field(default=0, title="ZMQ socket linger in milliseconds")
    enabled: bool = Field(default=False, title="Enable external model processing")


class ZmqModelRunner(ABC):
    """
    Base class for external model processing via ZMQ.
    
    Protocol:
    - Request is sent as a multipart message:
        [ header_json_bytes, data_bytes ]
      where header is a JSON object containing:
        {
          "model_type": str,  # ModelTypeEnum value
          "data_type": str,   # "tensor", "audio", "text", "image"
          "shape": List[int], # for tensor/audio data
          "dtype": str,       # numpy dtype string for tensor/audio
          "format": str,      # for image data: "jpg", "png", etc.
        }
      data_bytes are the raw bytes of the input data.

    - Response format depends on model type:
        - object_detection: tensor_bytes of shape (20, 6) float32
        - audio_detection: tensor_bytes of shape (20, 6) float32  
        - text_embedding: tensor_bytes of embedding vector
        - vision_embedding: tensor_bytes of embedding vector
        - face_recognition: tensor_bytes of feature vector
    """

    def __init__(self, config: ZmqModelConfig):
        self.config = config
        self._context = zmq.Context()
        self._socket = None
        self._create_socket()

    def _create_socket(self) -> None:
        """Create and configure ZMQ socket."""
        if self._socket is not None:
            try:
                self._socket.close(linger=self.config.linger_ms)
            except Exception:
                pass
                
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self.config.request_timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self.config.request_timeout_ms)
        self._socket.setsockopt(zmq.LINGER, self.config.linger_ms)
        
        logger.debug(f"External {self.config.model_type} model connecting to {self.config.endpoint}")
        self._socket.connect(self.config.endpoint)

    @abstractmethod
    def _build_header(self, input_data: Any) -> Dict[str, Any]:
        """Build header for the specific model type."""
        pass

    @abstractmethod
    def _decode_response(self, frames: List[bytes]) -> Any:
        """Decode response for the specific model type."""
        pass

    @abstractmethod
    def _get_fallback_result(self) -> Any:
        """Get fallback result on error."""
        pass

    def process(self, input_data: Any) -> Any:
        """Process input data with external model."""
        if not self.config.enabled:
            return self._get_fallback_result()
            
        try:
            header = self._build_header(input_data)
            header_bytes = json.dumps(header).encode("utf-8")
            
            if isinstance(input_data, np.ndarray):
                data_bytes = memoryview(input_data.tobytes(order="C"))
            elif isinstance(input_data, (str, bytes)):
                data_bytes = input_data.encode("utf-8") if isinstance(input_data, str) else input_data
            else:
                raise ValueError(f"Unsupported input data type: {type(input_data)}")
            
            # Send request
            self._socket.send_multipart([header_bytes, data_bytes])
            
            # Receive reply
            reply_frames = self._socket.recv_multipart()
            return self._decode_response(reply_frames)
            
        except zmq.Again:
            # Timeout
            logger.debug(f"External {self.config.model_type} model request timed out; resetting socket")
            try:
                self._create_socket()
            except Exception:
                pass
            return self._get_fallback_result()
            
        except zmq.ZMQError as exc:
            logger.error(f"External {self.config.model_type} model ZMQError: {exc}; resetting socket")
            try:
                self._create_socket()
            except Exception:
                pass
            return self._get_fallback_result()
            
        except Exception as exc:
            logger.error(f"External {self.config.model_type} model unexpected error: {exc}")
            return self._get_fallback_result()

    def __del__(self) -> None:
        """Cleanup ZMQ resources."""
        try:
            if self._socket is not None:
                self._socket.close(linger=self.config.linger_ms)
        except Exception:
            pass