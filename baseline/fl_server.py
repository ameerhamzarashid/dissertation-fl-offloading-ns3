#!/usr/bin/env python3
"""
Baseline Federated Learning Server
Simple server for edge computing offloading decisions
"""
import socket
import struct
import logging
import signal
import sys
from typing import Optional

# Configuration
HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 12345
TIMEOUT = 1.0

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("BaselineFL")

class SimpleFLServer:
    def __init__(self, host: str = HOST, port: int = PORT):
        self.host = host
        self.port = port
        self.running = False
        self.socket: Optional[socket.socket] = None
        
        # Simple offloading strategy based on state
        self.strategy_counter = 0
        
    def start(self):
        """Start the FL server"""
        self.running = True
        
        # Create and configure socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.settimeout(TIMEOUT)
        
        try:
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            logger.info(f"Baseline FL Server listening on {self.host}:{self.port}")
            
            while self.running:
                try:
                    conn, addr = self.socket.accept()
                    logger.info(f"Client connected: {addr}")
                    self.handle_client(conn, addr)
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        logger.error(f"Error accepting connection: {e}")
                        
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self.cleanup()
    
    def handle_client(self, conn: socket.socket, addr):
        """Handle individual client connections"""
        with conn:
            conn.settimeout(TIMEOUT)
            
            while self.running:
                try:
                    # Receive 4-byte float state
                    data = conn.recv(4)
                    if not data:
                        break
                        
                    if len(data) != 4:
                        logger.warning(f"Invalid data length from {addr}")
                        break
                    
                    # Parse network byte order float
                    network_state = struct.unpack('!I', data)[0]
                    state = struct.unpack('!f', struct.pack('!I', network_state))[0]
                    
                    logger.info(f"Received state: {state:.2f} from {addr}")
                    
                    # Compute offloading decision
                    action = self.compute_offloading_decision(state)
                    
                    # Send action back
                    network_action = struct.pack('!I', action)
                    conn.sendall(network_action)
                    
                    logger.info(f"Sent action: {action} to {addr}")
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"Error handling client {addr}: {e}")
                    break
                    
        logger.info(f"Client {addr} disconnected")
    
    def compute_offloading_decision(self, state: float) -> int:
        """
        Simple baseline offloading strategy
        
        Args:
            state: Current device state (0-100 representing resource utilization)
            
        Returns:
            action: Offloading decision (0=local, 1=edge, 2=cloud)
        """
        # Simple threshold-based strategy
        if state < 30.0:
            # Low utilization - process locally
            action = 0
        elif state < 70.0:
            # Medium utilization - offload to edge
            action = 1
        else:
            # High utilization - offload to cloud
            action = 2
            
        # Add some variation for testing
        self.strategy_counter += 1
        if self.strategy_counter % 10 == 0:
            action = (action + 1) % 3  # Rotate strategy occasionally
            
        return action
    
    def stop(self):
        """Stop the FL server"""
        logger.info("Stopping FL server...")
        self.running = False
        
    def cleanup(self):
        """Clean up resources"""
        if self.socket:
            self.socket.close()
        logger.info("FL server stopped")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal")
    if hasattr(signal_handler, 'server'):
        signal_handler.server.stop()
    sys.exit(0)

def main():
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start server
    server = SimpleFLServer()
    signal_handler.server = server
    
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server failed: {e}")
    finally:
        server.cleanup()

if __name__ == '__main__':
    main()
