#!/usr/bin/env python3
"""
Example external model server for Frigate.

This demonstrates how to create an external model server that can process
different types of models (object detection, audio detection, embeddings)
via ZMQ endpoints.

This example server would typically run on Apple Silicon or other external hardware
with optimized neural processing capabilities.
"""

import json
import logging
import signal
import sys
import threading
import time
from typing import Dict, Any

import numpy as np
import zmq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockModelServer:
    """Mock external model server for demonstration."""
    
    def __init__(self):
        self.context = zmq.Context()
        self.running = True
        self.sockets = {}
        
    def create_socket(self, endpoint: str, model_type: str):
        """Create a ZMQ socket for a specific model type."""
        socket = self.context.socket(zmq.REP)
        socket.bind(endpoint)
        self.sockets[model_type] = socket
        logger.info(f"Created {model_type} server at {endpoint}")
        
    def process_object_detection(self, header: Dict[str, Any], data: bytes) -> bytes:
        """Mock object detection processing."""
        shape = header.get("shape", [])
        dtype = header.get("dtype", "uint8")
        
        logger.info(f"Processing object detection: shape={shape}, dtype={dtype}")
        
        # Mock response: return zeros in the expected format (20, 6)
        result = np.zeros((20, 6), dtype=np.float32)
        # Add a mock detection
        result[0] = [1, 0.9, 0.1, 0.1, 0.9, 0.9]  # [class_id, confidence, x1, y1, x2, y2]
        
        return result.tobytes()
        
    def process_audio_detection(self, header: Dict[str, Any], data: bytes) -> bytes:
        """Mock audio detection processing."""
        shape = header.get("shape", [])
        dtype = header.get("dtype", "float32")
        
        logger.info(f"Processing audio detection: shape={shape}, dtype={dtype}")
        
        # Mock response: return zeros in the expected format (20, 6) 
        result = np.zeros((20, 6), dtype=np.float32)
        # Add a mock audio detection
        result[0] = [5, 0.8, -1, -1, -1, -1]  # [class_id, confidence, unused, unused, unused, unused]
        
        return result.tobytes()
        
    def process_text_embedding(self, header: Dict[str, Any], data: bytes) -> bytes:
        """Mock text embedding processing."""
        embedding_dim = header.get("embedding_dim", 768)
        batch_size = header.get("batch_size", 1)
        
        logger.info(f"Processing text embedding: dim={embedding_dim}, batch={batch_size}")
        
        # Mock response: return random embeddings
        result = np.random.randn(batch_size, embedding_dim).astype(np.float32)
        
        return result.tobytes()
        
    def process_vision_embedding(self, header: Dict[str, Any], data: bytes) -> bytes:
        """Mock vision embedding processing."""
        embedding_dim = header.get("embedding_dim", 768)
        batch_size = header.get("batch_size", 1)
        
        logger.info(f"Processing vision embedding: dim={embedding_dim}, batch={batch_size}")
        
        # Mock response: return random embeddings
        result = np.random.randn(batch_size, embedding_dim).astype(np.float32)
        
        return result.tobytes()
        
    def handle_request(self, model_type: str):
        """Handle requests for a specific model type."""
        socket = self.sockets[model_type]
        
        while self.running:
            try:
                # Receive request with timeout
                if socket.poll(timeout=1000):  # 1 second timeout
                    frames = socket.recv_multipart(zmq.NOBLOCK)
                    
                    if len(frames) >= 2:
                        header_data = frames[0]
                        payload_data = frames[1]
                        
                        try:
                            header = json.loads(header_data.decode('utf-8'))
                            request_model_type = header.get("model_type")
                            
                            logger.info(f"Received request for {request_model_type}")
                            
                            # Process based on model type
                            if request_model_type == "object_detection":
                                response = self.process_object_detection(header, payload_data)
                            elif request_model_type == "audio_detection":
                                response = self.process_audio_detection(header, payload_data)
                            elif request_model_type == "text_embedding":
                                response = self.process_text_embedding(header, payload_data)
                            elif request_model_type == "vision_embedding":
                                response = self.process_vision_embedding(header, payload_data)
                            else:
                                logger.warning(f"Unknown model type: {request_model_type}")
                                response = b""
                                
                            # Send response
                            socket.send(response)
                            
                        except Exception as e:
                            logger.error(f"Error processing request: {e}")
                            socket.send(b"")  # Send empty response on error
                    else:
                        logger.warning("Received malformed request")
                        socket.send(b"")
                        
            except zmq.Again:
                continue  # Timeout, continue loop
            except Exception as e:
                logger.error(f"Error in {model_type} handler: {e}")
                break
                
    def start(self):
        """Start the mock model server."""
        # Create sockets for different model types
        self.create_socket("ipc:///tmp/cache/zmq_detector", "object_detection")
        self.create_socket("ipc:///tmp/cache/zmq_audio", "audio_detection") 
        self.create_socket("ipc:///tmp/cache/zmq_text_embedding", "text_embedding")
        self.create_socket("ipc:///tmp/cache/zmq_vision_embedding", "vision_embedding")
        
        # Start handler threads
        threads = []
        for model_type in self.sockets.keys():
            thread = threading.Thread(target=self.handle_request, args=(model_type,))
            thread.daemon = True
            thread.start()
            threads.append(thread)
            
        logger.info("Mock external model server started")
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            
        self.running = False
        for thread in threads:
            thread.join(timeout=1)
            
        for socket in self.sockets.values():
            socket.close()
        self.context.term()
        
        logger.info("Mock external model server stopped")


def main():
    """Main entry point."""
    server = MockModelServer()
    
    def signal_handler(signum, frame):
        logger.info("Received signal, shutting down...")
        server.running = False
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    server.start()


if __name__ == "__main__":
    main()