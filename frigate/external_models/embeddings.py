"""ZMQ-based external embeddings processing."""

import json
import logging
from typing import Any, Dict, List

import numpy as np

from frigate.external_models.base import ModelTypeEnum, ZmqModelConfig, ZmqModelRunner

logger = logging.getLogger(__name__)


class ZmqTextEmbeddingConfig(ZmqModelConfig):
    """Configuration for ZMQ text embedding."""
    model_type: ModelTypeEnum = ModelTypeEnum.text_embedding
    embedding_dim: int = 768  # Default embedding dimension


class ZmqVisionEmbeddingConfig(ZmqModelConfig):
    """Configuration for ZMQ vision embedding."""
    model_type: ModelTypeEnum = ModelTypeEnum.vision_embedding
    embedding_dim: int = 768  # Default embedding dimension


class ZmqTextEmbedding(ZmqModelRunner):
    """
    ZMQ-based external text embedding processor.
    
    Compatible with existing text embedding interfaces but processes
    text through an external ZMQ endpoint.
    """

    def __init__(self, config: ZmqTextEmbeddingConfig):
        super().__init__(config)
        self.embedding_dim = config.embedding_dim
        self._zero_embedding = np.zeros(self.embedding_dim, dtype=np.float32)

    def _build_header(self, input_data: List[str]) -> Dict[str, Any]:
        """Build header for text embedding request."""
        return {
            "model_type": self.config.model_type.value,
            "data_type": "text",
            "batch_size": len(input_data),
            "embedding_dim": self.embedding_dim,
        }

    def _decode_response(self, frames: List[bytes]) -> np.ndarray:
        """Decode text embedding response."""
        try:
            if len(frames) == 1:
                # Single-frame raw embeddings
                buf = frames[0]
                expected_size = self.embedding_dim * 4  # float32
                if len(buf) % expected_size != 0:
                    logger.warning(
                        f"ZMQ text embedding received unexpected payload size: {len(buf)}"
                    )
                    return self._zero_embedding.reshape(1, -1)
                
                batch_size = len(buf) // expected_size
                return np.frombuffer(buf, dtype=np.float32).reshape((batch_size, self.embedding_dim))

            if len(frames) >= 2:
                header = json.loads(frames[0].decode("utf-8"))
                shape = tuple(header.get("shape", [1, self.embedding_dim]))
                dtype = np.dtype(header.get("dtype", "float32"))
                return np.frombuffer(frames[1], dtype=dtype).reshape(shape)

            logger.warning("ZMQ text embedding received empty reply")
            return self._zero_embedding.reshape(1, -1)
        except Exception as exc:
            logger.error(f"ZMQ text embedding failed to decode response: {exc}")
            return self._zero_embedding.reshape(1, -1)

    def _get_fallback_result(self) -> np.ndarray:
        """Get fallback result on error."""
        return self._zero_embedding.reshape(1, -1)

    def __call__(self, input_texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for input texts.
        
        Args:
            input_texts: List of strings to embed
            
        Returns:
            Embedding array of shape (batch_size, embedding_dim)
        """
        if not input_texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
            
        # Convert to JSON for transmission
        json_data = json.dumps(input_texts)
        return self.process(json_data)


class ZmqVisionEmbedding(ZmqModelRunner):
    """
    ZMQ-based external vision embedding processor.
    
    Compatible with existing vision embedding interfaces but processes
    images through an external ZMQ endpoint.
    """

    def __init__(self, config: ZmqVisionEmbeddingConfig):
        super().__init__(config)
        self.embedding_dim = config.embedding_dim
        self._zero_embedding = np.zeros(self.embedding_dim, dtype=np.float32)

    def _build_header(self, input_data: List[bytes]) -> Dict[str, Any]:
        """Build header for vision embedding request."""
        return {
            "model_type": self.config.model_type.value,
            "data_type": "image_batch",
            "batch_size": len(input_data),
            "embedding_dim": self.embedding_dim,
            "format": "jpg",  # Assuming JPEG format for thumbnails
        }

    def _decode_response(self, frames: List[bytes]) -> np.ndarray:
        """Decode vision embedding response."""
        try:
            if len(frames) == 1:
                # Single-frame raw embeddings
                buf = frames[0]
                expected_size = self.embedding_dim * 4  # float32
                if len(buf) % expected_size != 0:
                    logger.warning(
                        f"ZMQ vision embedding received unexpected payload size: {len(buf)}"
                    )
                    return self._zero_embedding.reshape(1, -1)
                
                batch_size = len(buf) // expected_size
                return np.frombuffer(buf, dtype=np.float32).reshape((batch_size, self.embedding_dim))

            if len(frames) >= 2:
                header = json.loads(frames[0].decode("utf-8"))
                shape = tuple(header.get("shape", [1, self.embedding_dim]))
                dtype = np.dtype(header.get("dtype", "float32"))
                return np.frombuffer(frames[1], dtype=dtype).reshape(shape)

            logger.warning("ZMQ vision embedding received empty reply")
            return self._zero_embedding.reshape(1, -1)
        except Exception as exc:
            logger.error(f"ZMQ vision embedding failed to decode response: {exc}")
            return self._zero_embedding.reshape(1, -1)

    def _get_fallback_result(self) -> np.ndarray:
        """Get fallback result on error."""
        return self._zero_embedding.reshape(1, -1)

    def __call__(self, input_images: List[bytes]) -> np.ndarray:
        """
        Generate embeddings for input images.
        
        Args:
            input_images: List of image bytes (JPEG format) to embed
            
        Returns:
            Embedding array of shape (batch_size, embedding_dim)
        """
        if not input_images:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
            
        # For multiple images, we need to package them appropriately
        # Create a simple format: [size1][image1][size2][image2]...
        packed_data = b""
        for img_bytes in input_images:
            size = len(img_bytes)
            packed_data += size.to_bytes(4, byteorder='little') + img_bytes
            
        return self.process(packed_data)