"""
NS-3 Interface - Python bridge to NS-3 network simulation
Handles communication between Python FL system and NS-3 simulation
"""

import subprocess
import socket
import json
import threading
import time
from typing import Dict, List, Any, Optional, Tuple
import os

from utils.logger import setup_logger


class NS3Interface:
    """
    Interface to communicate with NS-3 network simulation.
    Provides a bridge between Python federated learning system and NS-3.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize NS-3 interface.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = setup_logger("NS3Interface")
        
        # NS-3 simulation parameters
        self.ns3_script_path = config.get('ns3', {}).get('script_path', './ns3_modules/mec-environment')
        self.simulation_time = config.get('simulation', {}).get('duration', 600.0)
        
        # Communication parameters
        self.host = config.get('communication', {}).get('host', 'localhost')
        self.port = config.get('communication', {}).get('port', 9999)
        
        # Simulation state
        self.simulation_process = None
        self.is_running = False
        self.client_sockets = {}
        self.server_socket = None
        
        # Data collection
        self.network_metrics = []
        self.topology_info = {}
        
        self.logger.info("NS-3 Interface initialized")
    
    def start_simulation(self, scenario_config: Dict[str, Any]) -> bool:
        """
        Start NS-3 simulation with given scenario configuration.
        
        Args:
            scenario_config: Scenario-specific configuration
            
        Returns:
            True if simulation started successfully
        """
        try:
            # Prepare NS-3 command line arguments
            ns3_args = self._prepare_ns3_arguments(scenario_config)
            
            # Start NS-3 simulation process
            cmd = [self.ns3_script_path] + ns3_args
            self.simulation_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Start communication server for NS-3
            self._start_communication_server()
            
            self.is_running = True
            self.logger.info(f"NS-3 simulation started with PID: {self.simulation_process.pid}")
            
            # Wait for initial connection
            time.sleep(2.0)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start NS-3 simulation: {e}")
            return False
    
    def stop_simulation(self):
        """Stop the NS-3 simulation and cleanup resources."""
        self.is_running = False
        
        # Close communication server
        if self.server_socket:
            self.server_socket.close()
        
        # Terminate NS-3 process
        if self.simulation_process:
            self.simulation_process.terminate()
            try:
                self.simulation_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.simulation_process.kill()
                self.simulation_process.wait()
        
        # Close client connections
        for socket_conn in self.client_sockets.values():
            socket_conn.close()
        self.client_sockets.clear()
        
        self.logger.info("NS-3 simulation stopped")
    
    def _prepare_ns3_arguments(self, scenario_config: Dict[str, Any]) -> List[str]:
        """
        Prepare command line arguments for NS-3 simulation.
        
        Args:
            scenario_config: Scenario configuration
            
        Returns:
            List of command line arguments
        """
        args = []
        
        # Basic simulation parameters
        args.extend(['--duration', str(self.simulation_time)])
        args.extend(['--numUsers', str(scenario_config.get('num_users', 20))])
        args.extend(['--numServers', str(scenario_config.get('num_servers', 5))])
        
        # Network parameters
        network_config = self.config.get('network', {})
        args.extend(['--areaWidth', str(network_config.get('simulation_area', {}).get('width', 1000))])
        args.extend(['--areaHeight', str(network_config.get('simulation_area', {}).get('height', 1000))])
        args.extend(['--totalBandwidth', str(network_config.get('total_bandwidth', 20.0))])
        
        # Communication interface
        args.extend(['--serverHost', self.host])
        args.extend(['--serverPort', str(self.port)])
        
        # Output configuration
        output_dir = scenario_config.get('output_directory', 'ns3_results')
        args.extend(['--outputDir', output_dir])
        
        return args
    
    def _start_communication_server(self):
        """Start communication server for NS-3 clients."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(10)
            
            # Start accepting connections in a separate thread
            accept_thread = threading.Thread(target=self._accept_connections)
            accept_thread.daemon = True
            accept_thread.start()
            
            self.logger.info(f"Communication server started on {self.host}:{self.port}")
            
        except Exception as e:
            self.logger.error(f"Failed to start communication server: {e}")
            raise
    
    def _accept_connections(self):
        """Accept connections from NS-3 clients."""
        while self.is_running:
            try:
                client_socket, client_address = self.server_socket.accept()
                client_id = f"client_{len(self.client_sockets)}"
                
                self.client_sockets[client_id] = client_socket
                
                # Start client handler thread
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_id, client_socket)
                )
                client_thread.daemon = True
                client_thread.start()
                
                self.logger.info(f"NS-3 client connected: {client_address}")
                
            except Exception as e:
                if self.is_running:
                    self.logger.error(f"Error accepting connection: {e}")
                break
    
    def _handle_client(self, client_id: str, client_socket: socket.socket):
        """
        Handle communication with an NS-3 client.
        
        Args:
            client_id: Client identifier
            client_socket: Client socket connection
        """
        try:
            while self.is_running:
                # Receive data from NS-3 client
                data = client_socket.recv(4096)
                if not data:
                    break
                
                try:
                    message = json.loads(data.decode('utf-8'))
                    self._process_ns3_message(client_id, message)
                except json.JSONDecodeError:
                    self.logger.warning(f"Invalid JSON from NS-3 client {client_id}")
                
        except Exception as e:
            self.logger.error(f"Error handling NS-3 client {client_id}: {e}")
        finally:
            if client_id in self.client_sockets:
                del self.client_sockets[client_id]
            client_socket.close()
    
    def _process_ns3_message(self, client_id: str, message: Dict[str, Any]):
        """
        Process a message received from NS-3.
        
        Args:
            client_id: Client identifier
            message: Message dictionary
        """
        message_type = message.get('type', 'unknown')
        
        if message_type == 'topology_update':
            self._handle_topology_update(message)
        elif message_type == 'metrics_update':
            self._handle_metrics_update(message)
        elif message_type == 'task_request':
            self._handle_task_request(client_id, message)
        elif message_type == 'model_request':
            self._handle_model_request(client_id, message)
        else:
            self.logger.warning(f"Unknown message type from NS-3: {message_type}")
    
    def _handle_topology_update(self, message: Dict[str, Any]):
        """Handle network topology update from NS-3."""
        topology_data = message.get('data', {})
        self.topology_info.update(topology_data)
        
        self.logger.debug("Received topology update from NS-3")
    
    def _handle_metrics_update(self, message: Dict[str, Any]):
        """Handle network metrics update from NS-3."""
        metrics_data = message.get('data', {})
        metrics_data['timestamp'] = time.time()
        
        self.network_metrics.append(metrics_data)
        
        # Keep only recent metrics (last 1000 entries)
        if len(self.network_metrics) > 1000:
            self.network_metrics = self.network_metrics[-1000:]
    
    def _handle_task_request(self, client_id: str, message: Dict[str, Any]):
        """Handle task processing request from NS-3."""
        task_data = message.get('data', {})
        
        # This would integrate with the federated learning system
        # For now, send a simple acknowledgment
        response = {
            'type': 'task_response',
            'client_id': client_id,
            'task_id': task_data.get('task_id'),
            'status': 'accepted'
        }
        
        self._send_message_to_client(client_id, response)
    
    def _handle_model_request(self, client_id: str, message: Dict[str, Any]):
        """Handle model update request from NS-3."""
        # This would integrate with the federated learning coordinator
        # For now, send a placeholder response
        response = {
            'type': 'model_response',
            'client_id': client_id,
            'model_version': 1,
            'model_data': 'placeholder_model_data'
        }
        
        self._send_message_to_client(client_id, response)
    
    def _send_message_to_client(self, client_id: str, message: Dict[str, Any]):
        """
        Send a message to a specific NS-3 client.
        
        Args:
            client_id: Client identifier
            message: Message to send
        """
        if client_id in self.client_sockets:
            try:
                message_json = json.dumps(message)
                self.client_sockets[client_id].send(message_json.encode('utf-8'))
            except Exception as e:
                self.logger.error(f"Failed to send message to {client_id}: {e}")
    
    def broadcast_message(self, message: Dict[str, Any]):
        """
        Broadcast a message to all connected NS-3 clients.
        
        Args:
            message: Message to broadcast
        """
        for client_id in list(self.client_sockets.keys()):
            self._send_message_to_client(client_id, message)
    
    def get_network_topology(self) -> Dict[str, Any]:
        """
        Get current network topology information.
        
        Returns:
            Dictionary containing topology information
        """
        return self.topology_info.copy()
    
    def get_network_metrics(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get network performance metrics.
        
        Args:
            last_n: Number of recent metrics to return (None for all)
            
        Returns:
            List of metrics dictionaries
        """
        if last_n is None:
            return self.network_metrics.copy()
        else:
            return self.network_metrics[-last_n:].copy() if self.network_metrics else []
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """
        Get current simulation status.
        
        Returns:
            Dictionary containing simulation status
        """
        return {
            'is_running': self.is_running,
            'process_id': self.simulation_process.pid if self.simulation_process else None,
            'connected_clients': len(self.client_sockets),
            'metrics_collected': len(self.network_metrics),
            'simulation_time': self.simulation_time
        }
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for NS-3 simulation to complete.
        
        Args:
            timeout: Maximum time to wait (None for no timeout)
            
        Returns:
            True if simulation completed successfully
        """
        if not self.simulation_process:
            return False
        
        try:
            return_code = self.simulation_process.wait(timeout=timeout)
            self.is_running = False
            
            if return_code == 0:
                self.logger.info("NS-3 simulation completed successfully")
                return True
            else:
                self.logger.error(f"NS-3 simulation failed with return code: {return_code}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.warning("NS-3 simulation timeout")
            return False


class MockNS3Interface(NS3Interface):
    """
    Mock NS-3 interface for testing without actual NS-3 installation.
    Simulates NS-3 behavior for development and testing purposes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize mock interface."""
        super().__init__(config)
        self.logger.info("Using Mock NS-3 Interface for testing")
    
    def start_simulation(self, scenario_config: Dict[str, Any]) -> bool:
        """Start mock simulation."""
        self.is_running = True
        self.logger.info("Mock NS-3 simulation started")
        
        # Simulate some topology information
        self.topology_info = {
            'num_nodes': scenario_config.get('num_users', 20) + scenario_config.get('num_servers', 5),
            'area_size': [1000, 1000],
            'server_positions': [(200, 200), (800, 200), (500, 500), (200, 800), (800, 800)]
        }
        
        # Start generating mock metrics
        self._start_mock_metrics_generation()
        
        return True
    
    def _start_mock_metrics_generation(self):
        """Start generating mock network metrics."""
        def generate_metrics():
            while self.is_running:
                mock_metrics = {
                    'timestamp': time.time(),
                    'throughput_mbps': 15.0 + 10.0 * (0.5 - random.random()),
                    'latency_ms': 50.0 + 20.0 * random.random(),
                    'packet_loss_rate': 0.01 * random.random(),
                    'energy_consumption_joules': 1.0 + 0.5 * random.random()
                }
                self.network_metrics.append(mock_metrics)
                time.sleep(1.0)
        
        import random
        metrics_thread = threading.Thread(target=generate_metrics)
        metrics_thread.daemon = True
        metrics_thread.start()
    
    def stop_simulation(self):
        """Stop mock simulation."""
        self.is_running = False
        self.logger.info("Mock NS-3 simulation stopped")
