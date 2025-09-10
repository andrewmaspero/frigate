"""Tests for external models functionality."""

import numpy as np
import pytest

from frigate.external_models.base import ModelTypeEnum, ZmqModelConfig


def test_zmq_model_config():
    """Test basic ZmqModelConfig creation."""
    config = ZmqModelConfig(
        endpoint="ipc:///tmp/test",
        model_type=ModelTypeEnum.object_detection,
        request_timeout_ms=100,
        enabled=True,
    )
    
    assert config.endpoint == "ipc:///tmp/test"
    assert config.model_type == ModelTypeEnum.object_detection
    assert config.request_timeout_ms == 100
    assert config.enabled is True


def test_model_type_enum():
    """Test ModelTypeEnum values."""
    assert ModelTypeEnum.object_detection == "object_detection"
    assert ModelTypeEnum.audio_detection == "audio_detection"
    assert ModelTypeEnum.text_embedding == "text_embedding"
    assert ModelTypeEnum.vision_embedding == "vision_embedding"
    assert ModelTypeEnum.face_recognition == "face_recognition"


def test_audio_detection_config():
    """Test audio detection configuration."""
    from frigate.external_models.audio_detection import ZmqAudioDetectionConfig
    
    config = ZmqAudioDetectionConfig(
        endpoint="ipc:///tmp/audio",
        enabled=True,
    )
    
    assert config.model_type == ModelTypeEnum.audio_detection
    assert config.endpoint == "ipc:///tmp/audio"


def test_embeddings_config():
    """Test embeddings configuration."""
    from frigate.external_models.embeddings import ZmqTextEmbeddingConfig, ZmqVisionEmbeddingConfig
    
    text_config = ZmqTextEmbeddingConfig(
        endpoint="ipc:///tmp/text",
        embedding_dim=512,
        enabled=True,
    )
    
    vision_config = ZmqVisionEmbeddingConfig(
        endpoint="ipc:///tmp/vision",
        embedding_dim=1024,
        enabled=True,
    )
    
    assert text_config.model_type == ModelTypeEnum.text_embedding
    assert text_config.embedding_dim == 512
    assert vision_config.model_type == ModelTypeEnum.vision_embedding
    assert vision_config.embedding_dim == 1024


def test_external_models_config():
    """Test external models configuration."""
    from frigate.external_models.config import ExternalModelsConfig, ExternalAudioConfig, ExternalEmbeddingsConfig
    
    config = ExternalModelsConfig(
        audio=ExternalAudioConfig(
            enabled=True,
            endpoint="ipc:///tmp/audio_test"
        ),
        embeddings=ExternalEmbeddingsConfig(
            text_enabled=True,
            vision_enabled=True,
            text_endpoint="ipc:///tmp/text_test",
            vision_endpoint="ipc:///tmp/vision_test"
        )
    )
    
    assert config.audio.enabled is True
    assert config.audio.endpoint == "ipc:///tmp/audio_test"
    assert config.embeddings.text_enabled is True
    assert config.embeddings.vision_enabled is True


if __name__ == "__main__":
    # Run basic tests without pytest if needed
    test_zmq_model_config()
    test_model_type_enum()
    test_audio_detection_config()
    test_embeddings_config()
    test_external_models_config()
    print("All basic tests passed!")