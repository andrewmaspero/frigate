"""Configuration for external models."""

from typing import Optional

from pydantic import BaseModel, Field

from frigate.external_models.base import ModelTypeEnum


class ExternalAudioConfig(BaseModel):
    """Configuration for external audio detection."""
    enabled: bool = Field(default=False, title="Enable external audio detection")
    endpoint: str = Field(
        default="ipc:///tmp/cache/zmq_audio", title="ZMQ IPC endpoint for audio detection"
    )
    request_timeout_ms: int = Field(
        default=200, title="ZMQ request timeout in milliseconds"
    )
    linger_ms: int = Field(default=0, title="ZMQ socket linger in milliseconds")


class ExternalEmbeddingsConfig(BaseModel):
    """Configuration for external embeddings processing."""
    text_enabled: bool = Field(default=False, title="Enable external text embeddings")
    vision_enabled: bool = Field(default=False, title="Enable external vision embeddings")
    text_endpoint: str = Field(
        default="ipc:///tmp/cache/zmq_text_embedding", title="ZMQ IPC endpoint for text embeddings"
    )
    vision_endpoint: str = Field(
        default="ipc:///tmp/cache/zmq_vision_embedding", title="ZMQ IPC endpoint for vision embeddings"
    )
    text_embedding_dim: int = Field(default=768, title="Text embedding dimension")
    vision_embedding_dim: int = Field(default=768, title="Vision embedding dimension")
    request_timeout_ms: int = Field(
        default=1000, title="ZMQ request timeout in milliseconds"
    )
    linger_ms: int = Field(default=0, title="ZMQ socket linger in milliseconds")


class ExternalModelsConfig(BaseModel):
    """Configuration for all external model processing."""
    audio: Optional[ExternalAudioConfig] = Field(
        default_factory=ExternalAudioConfig, title="External audio detection configuration"
    )
    embeddings: Optional[ExternalEmbeddingsConfig] = Field(
        default_factory=ExternalEmbeddingsConfig, title="External embeddings configuration"
    )