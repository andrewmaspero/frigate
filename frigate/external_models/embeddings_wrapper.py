"""External embeddings wrapper for Frigate embeddings processing."""

import logging
from typing import List

import numpy as np

from frigate.external_models.base import ModelTypeEnum
from frigate.external_models.embeddings import (
    ZmqTextEmbedding,
    ZmqTextEmbeddingConfig,
    ZmqVisionEmbedding,
    ZmqVisionEmbeddingConfig,
)

logger = logging.getLogger(__name__)


class ExternalTextEmbedding:
    """
    Wrapper for external ZMQ text embedding that's compatible with existing text embedding interfaces.
    """

    def __init__(self, endpoint: str, embedding_dim: int = 768, request_timeout_ms: int = 1000, linger_ms: int = 0):
        config = ZmqTextEmbeddingConfig(
            endpoint=endpoint,
            model_type=ModelTypeEnum.text_embedding,
            embedding_dim=embedding_dim,
            request_timeout_ms=request_timeout_ms,
            linger_ms=linger_ms,
            enabled=True,
        )
        self.zmq_embedding = ZmqTextEmbedding(config)

    def __call__(self, input_texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for input texts.
        
        Args:
            input_texts: List of strings to embed
            
        Returns:
            Embedding array of shape (batch_size, embedding_dim)
        """
        return self.zmq_embedding(input_texts)


class ExternalVisionEmbedding:
    """
    Wrapper for external ZMQ vision embedding that's compatible with existing vision embedding interfaces.
    """

    def __init__(self, endpoint: str, embedding_dim: int = 768, request_timeout_ms: int = 1000, linger_ms: int = 0):
        config = ZmqVisionEmbeddingConfig(
            endpoint=endpoint,
            model_type=ModelTypeEnum.vision_embedding,
            embedding_dim=embedding_dim,
            request_timeout_ms=request_timeout_ms,
            linger_ms=linger_ms,
            enabled=True,
        )
        self.zmq_embedding = ZmqVisionEmbedding(config)

    def __call__(self, input_images: List[bytes]) -> np.ndarray:
        """
        Generate embeddings for input images.
        
        Args:
            input_images: List of image bytes (JPEG format) to embed
            
        Returns:
            Embedding array of shape (batch_size, embedding_dim)
        """
        return self.zmq_embedding(input_images)