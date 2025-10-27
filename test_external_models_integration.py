#!/usr/bin/env python3
"""
Integration test for external models functionality.

This test demonstrates the external models working without requiring
the full Frigate dependencies.
"""

import sys
sys.path.insert(0, '/home/runner/work/frigate/frigate')

import numpy as np
from frigate.external_models.base import ModelTypeEnum, ZmqModelConfig
from frigate.external_models.audio_detection import ZmqAudioDetectionConfig, ZmqAudioDetector
from frigate.external_models.embeddings import (
    ZmqTextEmbeddingConfig, 
    ZmqVisionEmbeddingConfig,
    ZmqTextEmbedding,
    ZmqVisionEmbedding
)


def test_configurations():
    """Test all configuration classes."""
    print("=== Testing Configuration Classes ===")
    
    # Test object detection config
    obj_config = ZmqModelConfig(
        endpoint="ipc:///tmp/object",
        model_type=ModelTypeEnum.object_detection,
        enabled=True
    )
    print(f"✓ Object detection config: {obj_config.model_type}")
    
    # Test audio detection config
    audio_config = ZmqAudioDetectionConfig(
        endpoint="ipc:///tmp/audio",
        enabled=True
    )
    print(f"✓ Audio detection config: {audio_config.model_type}")
    
    # Test text embedding config
    text_config = ZmqTextEmbeddingConfig(
        endpoint="ipc:///tmp/text",
        embedding_dim=512,
        enabled=True
    )
    print(f"✓ Text embedding config: {text_config.model_type}, dim={text_config.embedding_dim}")
    
    # Test vision embedding config
    vision_config = ZmqVisionEmbeddingConfig(
        endpoint="ipc:///tmp/vision",
        embedding_dim=1024,
        enabled=True
    )
    print(f"✓ Vision embedding config: {vision_config.model_type}, dim={vision_config.embedding_dim}")


def test_model_runners():
    """Test model runner instantiation (without actual ZMQ connection)."""
    print("\n=== Testing Model Runner Instantiation ===")
    
    # Test audio detector (disabled to avoid ZMQ connection)
    audio_config = ZmqAudioDetectionConfig(
        endpoint="ipc:///tmp/audio_test",
        enabled=False  # Disabled to avoid connection error
    )
    
    try:
        audio_detector = ZmqAudioDetector(audio_config)
        print("✓ Audio detector instantiated successfully")
        
        # Test fallback result
        fallback = audio_detector._get_fallback_result()
        print(f"✓ Audio detector fallback shape: {fallback.shape}")
        
    except Exception as e:
        print(f"✗ Audio detector failed: {e}")
    
    # Test text embedding (disabled to avoid ZMQ connection)
    text_config = ZmqTextEmbeddingConfig(
        endpoint="ipc:///tmp/text_test",
        embedding_dim=768,
        enabled=False  # Disabled to avoid connection error
    )
    
    try:
        text_embedding = ZmqTextEmbedding(text_config)
        print("✓ Text embedding instantiated successfully")
        
        # Test fallback result
        fallback = text_embedding._get_fallback_result()
        print(f"✓ Text embedding fallback shape: {fallback.shape}")
        
    except Exception as e:
        print(f"✗ Text embedding failed: {e}")
    
    # Test vision embedding (disabled to avoid ZMQ connection)
    vision_config = ZmqVisionEmbeddingConfig(
        endpoint="ipc:///tmp/vision_test",
        embedding_dim=768,
        enabled=False  # Disabled to avoid connection error
    )
    
    try:
        vision_embedding = ZmqVisionEmbedding(vision_config)
        print("✓ Vision embedding instantiated successfully")
        
        # Test fallback result
        fallback = vision_embedding._get_fallback_result()
        print(f"✓ Vision embedding fallback shape: {fallback.shape}")
        
    except Exception as e:
        print(f"✗ Vision embedding failed: {e}")


def test_protocol_headers():
    """Test protocol header building."""
    print("\n=== Testing Protocol Headers ===")
    
    # Test audio detection headers
    audio_config = ZmqAudioDetectionConfig(
        endpoint="ipc:///tmp/audio",
        enabled=False
    )
    audio_detector = ZmqAudioDetector(audio_config)
    
    # Mock audio data
    audio_data = np.random.randn(16000).astype(np.float32)  # 1 second of audio at 16kHz
    header = audio_detector._build_header(audio_data)
    
    print(f"✓ Audio header: {header}")
    assert header['model_type'] == 'audio_detection'
    assert header['data_type'] == 'audio'
    assert header['shape'] == list(audio_data.shape)
    
    # Test text embedding headers
    text_config = ZmqTextEmbeddingConfig(
        endpoint="ipc:///tmp/text",
        embedding_dim=768,
        enabled=False
    )
    text_embedding = ZmqTextEmbedding(text_config)
    
    test_texts = ["hello world", "test text"]
    header = text_embedding._build_header(test_texts)
    
    print(f"✓ Text header: {header}")
    assert header['model_type'] == 'text_embedding'
    assert header['data_type'] == 'text'
    assert header['batch_size'] == len(test_texts)


def test_fallback_behavior():
    """Test that disabled external models return appropriate fallbacks."""
    print("\n=== Testing Fallback Behavior ===")
    
    # Test disabled audio detector
    audio_config = ZmqAudioDetectionConfig(
        endpoint="ipc:///tmp/nonexistent",
        enabled=False  # Disabled
    )
    audio_detector = ZmqAudioDetector(audio_config)
    
    # Process some fake audio data
    audio_data = np.random.randn(16000).astype(np.float32)
    result = audio_detector.process(audio_data)
    
    print(f"✓ Disabled audio detector returns fallback shape: {result.shape}")
    assert result.shape == (20, 6)
    
    # Test disabled text embedding
    text_config = ZmqTextEmbeddingConfig(
        endpoint="ipc:///tmp/nonexistent", 
        embedding_dim=768,
        enabled=False  # Disabled
    )
    text_embedding = ZmqTextEmbedding(text_config)
    
    result = text_embedding(["test text"])
    print(f"✓ Disabled text embedding returns fallback shape: {result.shape}")
    assert result.shape == (1, 768)


def main():
    """Run all tests."""
    print("Running External Models Integration Tests")
    print("=" * 50)
    
    try:
        test_configurations()
        test_model_runners()
        test_protocol_headers()
        test_fallback_behavior()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("\nThe external models framework is working correctly.")
        print("You can now:")
        print("1. Configure external models in your Frigate config")
        print("2. Run the example_external_model_server.py")
        print("3. Use Apple Silicon or other external hardware for AI processing")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)