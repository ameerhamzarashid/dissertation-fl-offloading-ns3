#!/usr/bin/env python3
"""
Baseline FL Server with DQN and Replay Buffer
Federated learning server with DQN agent for intelligent offloading decisions
"""
import socket
import struct
import logging
import signal
import sys
import time
import threading
from typing import Optional
from simple_dqn import DQNAgent

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
logger = logging.getLogger("FL_DQN_Server")

class FLServerWithDQN:
    def __init__(self, host: str = HOST, port: int = PORT):
        self.host = host
        self.port = port
        self.running = False
        self.socket: Optional[socket.socket] = None
        
        # Initialize DQN agent
        self.dqn_agent = DQNAgent(
            state_dim=1,
            action_dim=3,
            lr=0.001,
            epsilon=0.1,  # Start with lower exploration for deployment
            use_prioritized=True  # Use prioritized replay buffer
        )
        
        # Training statistics
        self.total_interactions = 0
        self.training_episodes = 0
        self.client_states = {}  # Track previous states per client
        
    def start(self):
        """Start the FL server with DQN"""
        self.running = True
        
        # Create and configure socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.settimeout(TIMEOUT)
        
        try:
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            logger.info(f"FL Server with DQN listening on {self.host}:{self.port}")
            logger.info(f"DQN Agent: Prioritized Replay={self.dqn_agent.use_prioritized}")
            
            while self.running:
                try:
                    conn, addr = self.socket.accept()
                    logger.info(f"Client connected: {addr}")
                    
                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(conn, addr)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
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
        """Handle individual client connections with DQN"""
        client_id = f"{addr[0]}:{addr[1]}"
        
        with conn:
            conn.settimeout(TIMEOUT)
            interaction_count = 0
            
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
                    
                    interaction_count += 1
                    self.total_interactions += 1
                    
                    logger.info(f"Recv state: {state:.2f} from {addr} "
                              f"(interaction #{interaction_count})")
                    
                    # DQN-based offloading decision
                    action, reward = self.dqn_offloading_decision(client_id, state)
                    
                    # Train DQN if we have previous experience
                    if client_id in self.client_states:
                        prev_state, prev_action = self.client_states[client_id]
                        
                        # Calculate reward and store experience
                        computed_reward = self.dqn_agent.compute_reward(
                            prev_state, prev_action, state
                        )
                        
                        # Store experience in replay buffer
                        done = False  # Continuous learning
                        self.dqn_agent.remember(
                            prev_state, prev_action, computed_reward, state, done
                        )
                        
                        # Train if enough experiences
                        if len(self.dqn_agent.memory) > self.dqn_agent.batch_size:
                            self.dqn_agent.replay()
                            
                            if self.total_interactions % 100 == 0:
                                logger.info(f"DQN Training - Buffer size: {len(self.dqn_agent.memory)}, "
                                          f"Epsilon: {self.dqn_agent.epsilon:.3f}, "
                                          f"Training steps: {self.dqn_agent.training_step}")
                    
                    # Update client state
                    self.client_states[client_id] = (state, action)
                    
                    # Send action back
                    network_action = struct.pack('!I', action)
                    conn.sendall(network_action)
                    
                    logger.info(f"Sent action: {action} to {addr} "
                              f"(DQN epsilon: {self.dqn_agent.epsilon:.3f})")
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"Error handling client {addr}: {e}")
                    break
                    
        logger.info(f"Client {addr} disconnected ({interaction_count} interactions)")
        
        # Clean up client state
        if client_id in self.client_states:
            del self.client_states[client_id]
    
    def dqn_offloading_decision(self, client_id: str, state: float):
        """
        Make offloading decision using DQN agent
        
        Args:
            client_id: Unique client identifier
            state: Current device state (utilization)
            
        Returns:
            action: Offloading decision (0=local, 1=edge, 2=cloud)
            reward: Expected reward for this decision
        """
        # Normalize state to 0-100 range
        normalized_state = max(0, min(100, state))
        
        # Get action from DQN agent
        action = self.dqn_agent.act(normalized_state)
        
        # Calculate expected reward for monitoring
        expected_reward = self.dqn_agent.compute_reward(
            normalized_state, action, normalized_state
        )
        
        return action, expected_reward
    
    def save_model(self, filepath: str = "fl_dqn_model.pth"):
        """Save the trained DQN model"""
        try:
            self.dqn_agent.save(filepath)
            logger.info(f"DQN model saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load_model(self, filepath: str = "fl_dqn_model.pth"):
        """Load a pre-trained DQN model"""
        try:
            self.dqn_agent.load(filepath)
            logger.info(f"DQN model loaded from {filepath}")
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
    
    def stop(self):
        """Stop the FL server"""
        logger.info("Stopping FL server...")
        self.running = False
        
        # Save model before shutdown
        self.save_model()
        
    def cleanup(self):
        """Clean up resources"""
        if self.socket:
            self.socket.close()
        logger.info("FL server stopped")
        logger.info(f"Total interactions processed: {self.total_interactions}")
        logger.info(f"Final DQN buffer size: {len(self.dqn_agent.memory)}")

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
    server = FLServerWithDQN()
    signal_handler.server = server
    
    # Try to load existing model
    server.load_model()
    
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
