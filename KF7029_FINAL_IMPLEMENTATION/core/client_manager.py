"""
Client Manager - Handles client lifecycle and communication
Manages the registration, communication, and lifecycle of FL clients
"""

import threading
import queue
import socket
import json
import time
from typing import Dict, List, Any, Optional, Tuple
import logging

from utils.logger import setup_logger


class ClientManager:
    """
    Manages federated learning clients including communication and coordination.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the client manager.
        
        Args:
            config: System configuration dictionary
        """
        self.config = config
        self.logger = setup_logger("ClientManager")
        
        # Client management
        self.active_clients = {}
        self.client_status = {}
        self.client_performance_history = {}
        
        # Communication setup
        self.server_host = config.get('communication', {}).get('host', 'localhost')
        self.server_port = config.get('communication', {}).get('port', 8888)
        self.max_clients = config.get('network', {}).get('num_mobile_users', 10)
        
        # Threading for concurrent client handling
        self.client_threads = {}
        self.message_queues = {}
        self.lock = threading.Lock()
        
        # Server socket for client connections
        self.server_socket = None
        self.is_running = False
        
        self.logger.info(f"Client Manager initialized for {self.max_clients} clients")
    
    def start_server(self):
        """
        Start the server to accept client connections.
        """
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.server_host, self.server_port))
            self.server_socket.listen(self.max_clients)
            
            self.is_running = True
            self.logger.info(f"Server started on {self.server_host}:{self.server_port}")
            
            # Start accepting connections in a separate thread
            accept_thread = threading.Thread(target=self._accept_connections)
            accept_thread.daemon = True
            accept_thread.start()
            
        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            raise
    
    def stop_server(self):
        """
        Stop the server and close all client connections.
        """
        self.is_running = False
        
        if self.server_socket:
            self.server_socket.close()
        
        # Close all client connections
        with self.lock:
            for client_id in list(self.active_clients.keys()):
                self._disconnect_client(client_id)
        
        self.logger.info("Server stopped")
    
    def _accept_connections(self):
        """
        Accept incoming client connections.
        """
        while self.is_running:
            try:
                client_socket, client_address = self.server_socket.accept()
                client_id = self._generate_client_id()
                
                with self.lock:
                    self.active_clients[client_id] = {
                        'socket': client_socket,
                        'address': client_address,
                        'connected_at': time.time(),
                        'last_activity': time.time()
                    }
                    
                    self.client_status[client_id] = 'connected'
                    self.client_performance_history[client_id] = []
                    self.message_queues[client_id] = queue.Queue()
                
                # Start client handler thread
                client_thread = threading.Thread(
                    target=self._handle_client, 
                    args=(client_id,)
                )
                client_thread.daemon = True
                client_thread.start()
                
                self.client_threads[client_id] = client_thread
                
                self.logger.info(f"Client {client_id} connected from {client_address}")
                
            except Exception as e:
                if self.is_running:
                    self.logger.error(f"Error accepting connection: {e}")
                break
    
    def _generate_client_id(self) -> str:
        """
        Generate a unique client ID.
        
        Returns:
            Unique client identifier string
        """
        timestamp = int(time.time() * 1000)
        client_count = len(self.active_clients)
        return f"client_{client_count}_{timestamp}"
    
    def _handle_client(self, client_id: str):
        """
        Handle communication with a specific client.
        
        Args:
            client_id: Unique client identifier
        """
        client_info = self.active_clients[client_id]
        client_socket = client_info['socket']
        
        try:
            while self.is_running and client_id in self.active_clients:
                # Check for incoming messages
                try:
                    client_socket.settimeout(1.0)  # Non-blocking with timeout
                    data = client_socket.recv(4096)
                    
                    if not data:
                        break
                    
                    # Process received message
                    message = json.loads(data.decode('utf-8'))
                    self._process_client_message(client_id, message)
                    
                    # Update last activity
                    with self.lock:
                        self.active_clients[client_id]['last_activity'] = time.time()
                    
                except socket.timeout:
                    continue
                except json.JSONDecodeError:
                    self.logger.warning(f"Invalid JSON from client {client_id}")
                    continue
                
                # Send pending messages to client
                self._send_pending_messages(client_id)
                
        except Exception as e:
            self.logger.error(f"Error handling client {client_id}: {e}")
        finally:
            self._disconnect_client(client_id)
    
    def _process_client_message(self, client_id: str, message: Dict[str, Any]):
        """
        Process a message received from a client.
        
        Args:
            client_id: Client identifier
            message: Received message dictionary
        """
        message_type = message.get('type', 'unknown')
        
        if message_type == 'registration':
            self._handle_registration(client_id, message)
        elif message_type == 'model_update':
            self._handle_model_update(client_id, message)
        elif message_type == 'training_complete':
            self._handle_training_complete(client_id, message)
        elif message_type == 'heartbeat':
            self._handle_heartbeat(client_id, message)
        else:
            self.logger.warning(f"Unknown message type '{message_type}' from client {client_id}")
    
    def _handle_registration(self, client_id: str, message: Dict[str, Any]):
        """
        Handle client registration message.
        
        Args:
            client_id: Client identifier
            message: Registration message
        """
        client_info = message.get('client_info', {})
        
        with self.lock:
            self.active_clients[client_id].update({
                'device_type': client_info.get('device_type', 'unknown'),
                'capabilities': client_info.get('capabilities', {}),
                'battery_level': client_info.get('battery_level', 100)
            })
            
            self.client_status[client_id] = 'registered'
        
        # Send registration confirmation
        response = {
            'type': 'registration_response',
            'status': 'success',
            'assigned_id': client_id,
            'server_config': self.config.get('client_config', {})
        }
        
        self._queue_message(client_id, response)
        self.logger.info(f"Client {client_id} registered successfully")
    
    def _handle_model_update(self, client_id: str, message: Dict[str, Any]):
        """
        Handle model update from client.
        
        Args:
            client_id: Client identifier
            message: Model update message
        """
        update_data = message.get('update_data', {})
        training_stats = message.get('training_stats', {})
        
        # Store update for aggregation
        with self.lock:
            self.client_status[client_id] = 'update_received'
            
            # Track performance
            performance_score = training_stats.get('average_reward', 0.0)
            self.client_performance_history[client_id].append({
                'timestamp': time.time(),
                'performance': performance_score,
                'training_stats': training_stats
            })
        
        self.logger.debug(f"Received model update from client {client_id}")
    
    def _handle_training_complete(self, client_id: str, message: Dict[str, Any]):
        """
        Handle training completion notification.
        
        Args:
            client_id: Client identifier
            message: Training completion message
        """
        training_results = message.get('results', {})
        
        with self.lock:
            self.client_status[client_id] = 'training_complete'
        
        self.logger.info(f"Client {client_id} completed training")
    
    def _handle_heartbeat(self, client_id: str, message: Dict[str, Any]):
        """
        Handle heartbeat message from client.
        
        Args:
            client_id: Client identifier
            message: Heartbeat message
        """
        # Update client status and last activity
        with self.lock:
            if client_id in self.active_clients:
                self.active_clients[client_id]['last_activity'] = time.time()
                
                # Update battery level if provided
                battery_level = message.get('battery_level')
                if battery_level is not None:
                    self.active_clients[client_id]['battery_level'] = battery_level
    
    def _send_pending_messages(self, client_id: str):
        """
        Send all pending messages to a client.
        
        Args:
            client_id: Client identifier
        """
        if client_id not in self.message_queues:
            return
        
        message_queue = self.message_queues[client_id]
        client_socket = self.active_clients[client_id]['socket']
        
        try:
            while not message_queue.empty():
                message = message_queue.get_nowait()
                message_json = json.dumps(message)
                client_socket.send(message_json.encode('utf-8'))
                
        except queue.Empty:
            pass
        except Exception as e:
            self.logger.error(f"Error sending message to client {client_id}: {e}")
    
    def _queue_message(self, client_id: str, message: Dict[str, Any]):
        """
        Queue a message to be sent to a client.
        
        Args:
            client_id: Client identifier
            message: Message to queue
        """
        if client_id in self.message_queues:
            self.message_queues[client_id].put(message)
    
    def _disconnect_client(self, client_id: str):
        """
        Disconnect a client and clean up resources.
        
        Args:
            client_id: Client identifier
        """
        with self.lock:
            if client_id in self.active_clients:
                try:
                    self.active_clients[client_id]['socket'].close()
                except:
                    pass
                
                del self.active_clients[client_id]
                self.client_status[client_id] = 'disconnected'
            
            if client_id in self.message_queues:
                del self.message_queues[client_id]
            
            if client_id in self.client_threads:
                del self.client_threads[client_id]
        
        self.logger.info(f"Client {client_id} disconnected")
    
    def broadcast_message(self, message: Dict[str, Any], target_clients: Optional[List[str]] = None):
        """
        Broadcast a message to all or specified clients.
        
        Args:
            message: Message to broadcast
            target_clients: List of target client IDs (None for all clients)
        """
        if target_clients is None:
            target_clients = list(self.active_clients.keys())
        
        for client_id in target_clients:
            if client_id in self.active_clients:
                self._queue_message(client_id, message)
        
        self.logger.debug(f"Broadcast message to {len(target_clients)} clients")
    
    def get_active_clients(self) -> List[str]:
        """
        Get list of currently active client IDs.
        
        Returns:
            List of active client identifiers
        """
        with self.lock:
            return list(self.active_clients.keys())
    
    def get_client_status(self, client_id: str) -> Optional[str]:
        """
        Get the status of a specific client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Client status string or None if client not found
        """
        return self.client_status.get(client_id)
    
    def get_client_performance_history(self, client_id: str) -> List[Dict[str, Any]]:
        """
        Get performance history for a specific client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            List of performance records
        """
        return self.client_performance_history.get(client_id, [])
    
    def get_client_statistics(self) -> Dict[str, Any]:
        """
        Get overall client statistics.
        
        Returns:
            Dictionary containing client statistics
        """
        with self.lock:
            total_clients = len(self.active_clients)
            connected_clients = sum(1 for status in self.client_status.values() 
                                  if status == 'connected')
            registered_clients = sum(1 for status in self.client_status.values() 
                                   if status == 'registered')
            
            avg_battery = 0.0
            if self.active_clients:
                battery_levels = [client.get('battery_level', 100) 
                                for client in self.active_clients.values()]
                avg_battery = sum(battery_levels) / len(battery_levels)
        
        return {
            'total_clients': total_clients,
            'connected_clients': connected_clients,
            'registered_clients': registered_clients,
            'average_battery_level': avg_battery,
            'uptime': time.time() - self.config.get('start_time', time.time())
        }
