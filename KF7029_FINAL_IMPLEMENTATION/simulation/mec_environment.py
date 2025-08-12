"""
MEC Environment Simulation
Simulates the Mobile Edge Computing environment for task offloading decisions
"""

import numpy as np
import random
from typing import Tuple, Dict, List, Optional, Any
import time
from dataclasses import dataclass
from enum import Enum

from utils.logger import setup_logger


class TaskType(Enum):
    """Enumeration for different types of computational tasks."""
    COMPUTATION_INTENSIVE = "computation"
    DATA_INTENSIVE = "data"
    LATENCY_SENSITIVE = "latency"
    MIXED = "mixed"


@dataclass
class Task:
    """Represents a computational task to be processed."""
    task_id: str
    data_size: float  # MB
    computational_complexity: float  # Megacycles
    deadline: float  # seconds
    task_type: TaskType
    arrival_time: float
    priority: float = 1.0


@dataclass
class EdgeServer:
    """Represents an edge server with processing capabilities."""
    server_id: int
    cpu_frequency: float  # GHz
    memory: float  # GB
    current_load: float  # 0.0 to 1.0
    position: Tuple[float, float]  # (x, y) coordinates
    energy_efficiency: float  # Tasks per Joule
    queue_length: int = 0
    processing_queue: List[Task] = None
    
    def __post_init__(self):
        if self.processing_queue is None:
            self.processing_queue = []


@dataclass
class MobileUser:
    """Represents a mobile user device."""
    user_id: int
    position: Tuple[float, float]
    velocity: Tuple[float, float]  # (vx, vy) m/s
    cpu_frequency: float  # GHz
    battery_level: float  # 0.0 to 1.0
    memory: float  # GB
    task_queue: List[Task] = None
    
    def __post_init__(self):
        if self.task_queue is None:
            self.task_queue = []


class MECEnvironment:
    """
    Mobile Edge Computing environment simulator for federated learning.
    Provides realistic simulation of task generation, processing, and offloading decisions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MEC environment.
        
        Args:
            config: Configuration dictionary containing simulation parameters
        """
        self.config = config
        self.logger = setup_logger("MECEnvironment")
        
        # Environment parameters
        self.area_width = config['network']['simulation_area']['width']
        self.area_height = config['network']['simulation_area']['height']
        self.num_edge_servers = config['network']['num_edge_servers']
        self.num_mobile_users = config['network']['num_mobile_users']
        
        # Task generation parameters
        self.task_arrival_rate = config['task_generation']['arrival_rate']
        self.data_size_range = (
            config['task_generation']['data_size']['min'],
            config['task_generation']['data_size']['max']
        )
        self.complexity_range = (
            config['task_generation']['computational_complexity']['min'],
            config['task_generation']['computational_complexity']['max']
        )
        
        # Device capabilities
        self.mobile_cpu_freq = config['device_capabilities']['mobile_user']['cpu_frequency']
        self.edge_cpu_freq = config['device_capabilities']['edge_server']['cpu_frequency']
        
        # Communication parameters
        self.total_bandwidth = config['network']['total_bandwidth']  # MHz
        self.transmission_power = config['network']['transmission_power']  # Watts
        
        # Initialize environment components
        self.edge_servers = []
        self.mobile_users = []
        self.current_time = 0.0
        self.task_counter = 0
        
        # Performance tracking
        self.completed_tasks = []
        self.dropped_tasks = []
        self.energy_consumption_history = []
        self.latency_history = []
        
        # State space and action space definitions
        self.state_dim = config['learning']['state_dim']
        self.action_dim = config['learning']['action_dim']
        
        self._initialize_infrastructure()
        self._initialize_mobile_users()
        
        self.logger.info(f"MEC Environment initialized with {self.num_edge_servers} servers "
                        f"and {self.num_mobile_users} mobile users")
    
    def _initialize_infrastructure(self):
        """Initialize edge servers at fixed positions."""
        # Place edge servers in a grid pattern for coverage
        positions = [
            (200, 200), (800, 200), (500, 500), (200, 800), (800, 800)
        ]
        
        for i in range(self.num_edge_servers):
            if i < len(positions):
                position = positions[i]
            else:
                # Random placement for additional servers
                position = (
                    random.uniform(100, self.area_width - 100),
                    random.uniform(100, self.area_height - 100)
                )
            
            server = EdgeServer(
                server_id=i,
                cpu_frequency=self.edge_cpu_freq,
                memory=32.0,  # GB
                current_load=0.0,
                position=position,
                energy_efficiency=100.0  # Tasks per Joule
            )
            
            self.edge_servers.append(server)
    
    def _initialize_mobile_users(self):
        """Initialize mobile users with random positions and characteristics."""
        for i in range(self.num_mobile_users):
            # Random initial position
            position = (
                random.uniform(50, self.area_width - 50),
                random.uniform(50, self.area_height - 50)
            )
            
            # Random initial velocity (Gaussian-Markov will update this)
            velocity = (
                random.uniform(-2.0, 2.0),  # m/s
                random.uniform(-2.0, 2.0)   # m/s
            )
            
            user = MobileUser(
                user_id=i,
                position=position,
                velocity=velocity,
                cpu_frequency=self.mobile_cpu_freq,
                battery_level=random.uniform(0.5, 1.0),  # 50-100% battery
                memory=4.0  # GB
            )
            
            self.mobile_users.append(user)
    
    def reset(self, user_id: int = 0) -> np.ndarray:
        """
        Reset the environment for a specific user.
        
        Args:
            user_id: ID of the mobile user
            
        Returns:
            Initial state vector
        """
        # Reset user's task queue
        if user_id < len(self.mobile_users):
            self.mobile_users[user_id].task_queue.clear()
        
        # Reset server loads
        for server in self.edge_servers:
            server.current_load = random.uniform(0.1, 0.5)  # Initial load
            server.queue_length = random.randint(0, 5)
        
        # Reset time
        self.current_time = 0.0
        
        # Generate initial state
        state = self._get_state(user_id)
        
        return state
    
    def step(self, action: int, user_id: int = 0) -> Tuple[np.ndarray, float, bool]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action taken by the agent (0=local, 1-5=offload to server)
            user_id: ID of the mobile user
            
        Returns:
            Tuple of (next_state, reward, done)
        """
        if user_id >= len(self.mobile_users):
            raise ValueError(f"Invalid user_id: {user_id}")
        
        user = self.mobile_users[user_id]
        
        # Generate a new task for this step
        task = self._generate_task(user_id)
        user.task_queue.append(task)
        
        # Process the offloading decision
        reward = self._process_offloading_decision(user, task, action)
        
        # Update environment state
        self._update_environment()
        
        # Get next state
        next_state = self._get_state(user_id)
        
        # Episode termination (simplified - could be based on battery, time, etc.)
        done = len(self.completed_tasks) >= 100 or user.battery_level <= 0.1
        
        return next_state, reward, done
    
    def _generate_task(self, user_id: int) -> Task:
        """
        Generate a new computational task using Poisson arrival process.
        
        Args:
            user_id: ID of the user generating the task
            
        Returns:
            Generated task
        """
        # Task characteristics based on uniform distributions
        data_size = random.uniform(*self.data_size_range)
        complexity = random.uniform(*self.complexity_range)
        
        # Task deadline based on complexity (more complex = longer deadline)
        deadline = 5.0 + (complexity / 1000.0) * 10.0  # 5-15 seconds
        
        # Task type distribution
        task_types = list(TaskType)
        task_type = random.choice(task_types)
        
        task = Task(
            task_id=f"task_{user_id}_{self.task_counter}",
            data_size=data_size,
            computational_complexity=complexity,
            deadline=deadline,
            task_type=task_type,
            arrival_time=self.current_time,
            priority=random.uniform(0.5, 1.0)
        )
        
        self.task_counter += 1
        return task
    
    def _process_offloading_decision(self, user: MobileUser, task: Task, action: int) -> float:
        """
        Process the offloading decision and calculate reward.
        
        Args:
            user: Mobile user making the decision
            task: Task to be processed
            action: Offloading decision (0=local, 1-5=server)
            
        Returns:
            Reward value
        """
        if action == 0:
            # Local processing
            return self._process_locally(user, task)
        elif 1 <= action <= self.num_edge_servers:
            # Offload to edge server
            server_id = action - 1
            return self._process_remotely(user, task, self.edge_servers[server_id])
        else:
            # Invalid action
            self.logger.warning(f"Invalid action: {action}")
            return -10.0  # Penalty for invalid action
    
    def _process_locally(self, user: MobileUser, task: Task) -> float:
        """
        Process task locally on the mobile device.
        
        Args:
            user: Mobile user
            task: Task to process
            
        Returns:
            Reward for local processing
        """
        # Calculate processing time
        processing_time = task.computational_complexity / (user.cpu_frequency * 1000)  # seconds
        
        # Calculate energy consumption (simplified model)
        processing_power = 2.0  # Watts during computation
        energy_consumed = processing_power * processing_time
        
        # Update battery level
        battery_drain = energy_consumed / 10000.0  # Simplified battery model
        user.battery_level = max(0.0, user.battery_level - battery_drain)
        
        # Calculate latency
        total_latency = processing_time
        
        # Check if task meets deadline
        if total_latency <= task.deadline:
            self.completed_tasks.append({
                'task_id': task.task_id,
                'processing_location': 'local',
                'latency': total_latency,
                'energy': energy_consumed,
                'success': True
            })
            
            # Reward calculation (negative because we want to minimize)
            latency_penalty = total_latency / task.deadline
            energy_penalty = energy_consumed / 10.0
            reward = -(latency_penalty + energy_penalty)
        else:
            # Task deadline missed
            self.dropped_tasks.append(task.task_id)
            reward = -5.0  # Large penalty for missing deadline
        
        # Track metrics
        self.latency_history.append(total_latency)
        self.energy_consumption_history.append(energy_consumed)
        
        return reward
    
    def _process_remotely(self, user: MobileUser, task: Task, server: EdgeServer) -> float:
        """
        Process task on an edge server.
        
        Args:
            user: Mobile user
            task: Task to process
            server: Target edge server
            
        Returns:
            Reward for remote processing
        """
        # Calculate transmission time
        distance = self._calculate_distance(user.position, server.position)
        channel_rate = self._calculate_channel_rate(distance)  # bps
        transmission_time = (task.data_size * 8 * 1024 * 1024) / channel_rate  # seconds
        
        # Calculate processing time on server (faster CPU)
        base_processing_time = task.computational_complexity / (server.cpu_frequency * 1000)
        # Add queuing delay based on server load
        queuing_delay = server.current_load * base_processing_time
        processing_time = base_processing_time + queuing_delay
        
        # Calculate energy consumption (transmission only for user)
        transmission_energy = self.transmission_power * transmission_time
        
        # Update battery level
        battery_drain = transmission_energy / 10000.0  # Simplified battery model
        user.battery_level = max(0.0, user.battery_level - battery_drain)
        
        # Update server load
        server.current_load = min(1.0, server.current_load + 0.1)
        server.queue_length += 1
        
        # Calculate total latency (transmission + processing + return transmission)
        return_transmission_time = transmission_time * 0.1  # Result is much smaller
        total_latency = transmission_time + processing_time + return_transmission_time
        
        # Check if task meets deadline
        if total_latency <= task.deadline:
            self.completed_tasks.append({
                'task_id': task.task_id,
                'processing_location': f'server_{server.server_id}',
                'latency': total_latency,
                'energy': transmission_energy,
                'success': True
            })
            
            # Reward calculation
            latency_penalty = total_latency / task.deadline
            energy_penalty = transmission_energy / 5.0  # Lower energy penalty for offloading
            communication_penalty = transmission_time / 10.0  # Penalty for communication delay
            reward = -(latency_penalty + energy_penalty + communication_penalty)
        else:
            # Task deadline missed
            self.dropped_tasks.append(task.task_id)
            reward = -5.0  # Large penalty for missing deadline
        
        # Track metrics
        self.latency_history.append(total_latency)
        self.energy_consumption_history.append(transmission_energy)
        
        return reward
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _calculate_channel_rate(self, distance: float) -> float:
        """
        Calculate wireless channel data rate based on distance.
        
        Args:
            distance: Distance in meters
            
        Returns:
            Data rate in bits per second
        """
        # Path loss calculation
        path_loss_db = 40 + 35 * np.log10(distance)  # Free space path loss
        path_loss_linear = 10 ** (path_loss_db / 10)
        
        # Received power
        tx_power_dbm = 23  # 200 mW
        tx_power_watts = 10 ** ((tx_power_dbm - 30) / 10)
        rx_power_watts = tx_power_watts / path_loss_linear
        
        # Signal-to-noise ratio
        noise_power_dbm = -174 + 10 * np.log10(self.total_bandwidth * 1e6)  # dBm
        noise_power_watts = 10 ** ((noise_power_dbm - 30) / 10)
        snr = rx_power_watts / noise_power_watts
        
        # Shannon capacity
        channel_rate = self.total_bandwidth * 1e6 * np.log2(1 + snr)  # bps
        
        return max(channel_rate, 1e6)  # Minimum 1 Mbps
    
    def _get_state(self, user_id: int) -> np.ndarray:
        """
        Get current state vector for a mobile user.
        
        Args:
            user_id: ID of the mobile user
            
        Returns:
            State vector as numpy array
        """
        user = self.mobile_users[user_id]
        
        # State components:
        # [0] Task queue length (normalized)
        # [1] Battery level
        # [2-6] Latency to each edge server (normalized)
        # [7-11] Load indicator for each edge server
        
        state = np.zeros(self.state_dim)
        
        # Task queue length (normalized by max expected queue size)
        state[0] = min(len(user.task_queue) / 10.0, 1.0)
        
        # Battery level
        state[1] = user.battery_level
        
        # Latency to each edge server
        for i, server in enumerate(self.edge_servers):
            distance = self._calculate_distance(user.position, server.position)
            # Normalize distance by diagonal of area
            max_distance = np.sqrt(self.area_width**2 + self.area_height**2)
            normalized_distance = distance / max_distance
            state[2 + i] = normalized_distance
        
        # Server load indicators
        for i, server in enumerate(self.edge_servers):
            state[7 + i] = server.current_load
        
        return state
    
    def _update_environment(self):
        """Update environment state (mobility, server loads, etc.)."""
        # Update time
        self.current_time += 1.0  # 1 second time step
        
        # Update user mobility (simplified Gaussian-Markov)
        for user in self.mobile_users:
            self._update_user_mobility(user)
        
        # Update server loads (decay over time)
        for server in self.edge_servers:
            server.current_load = max(0.0, server.current_load - 0.05)
            server.queue_length = max(0, server.queue_length - 1)
    
    def _update_user_mobility(self, user: MobileUser):
        """
        Update user position using simplified Gaussian-Markov mobility.
        
        Args:
            user: Mobile user to update
        """
        # Gaussian-Markov mobility parameters
        alpha = 0.5  # Memory parameter
        dt = 1.0     # Time step in seconds
        
        # Update velocity (with memory and random component)
        vx_new = alpha * user.velocity[0] + (1 - alpha) * random.gauss(0, 2.0)
        vy_new = alpha * user.velocity[1] + (1 - alpha) * random.gauss(0, 2.0)
        
        # Limit velocity
        max_velocity = 10.0  # m/s
        vx_new = max(-max_velocity, min(max_velocity, vx_new))
        vy_new = max(-max_velocity, min(max_velocity, vy_new))
        
        user.velocity = (vx_new, vy_new)
        
        # Update position
        new_x = user.position[0] + vx_new * dt
        new_y = user.position[1] + vy_new * dt
        
        # Boundary handling (reflection)
        if new_x < 0 or new_x > self.area_width:
            vx_new = -vx_new
            new_x = max(0, min(self.area_width, new_x))
        
        if new_y < 0 or new_y > self.area_height:
            vy_new = -vy_new
            new_y = max(0, min(self.area_height, new_y))
        
        user.position = (new_x, new_y)
        user.velocity = (vx_new, vy_new)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the simulation.
        
        Returns:
            Dictionary containing performance metrics
        """
        if not self.completed_tasks:
            return {'status': 'No completed tasks'}
        
        # Calculate metrics
        avg_latency = np.mean(self.latency_history) if self.latency_history else 0.0
        avg_energy = np.mean(self.energy_consumption_history) if self.energy_consumption_history else 0.0
        
        success_rate = len(self.completed_tasks) / (len(self.completed_tasks) + len(self.dropped_tasks))
        
        # Local vs remote processing statistics
        local_tasks = [t for t in self.completed_tasks if t['processing_location'] == 'local']
        remote_tasks = [t for t in self.completed_tasks if t['processing_location'] != 'local']
        
        metrics = {
            'total_completed_tasks': len(self.completed_tasks),
            'total_dropped_tasks': len(self.dropped_tasks),
            'success_rate': success_rate,
            'average_latency': avg_latency,
            'average_energy_consumption': avg_energy,
            'local_processing_ratio': len(local_tasks) / len(self.completed_tasks) if self.completed_tasks else 0.0,
            'remote_processing_ratio': len(remote_tasks) / len(self.completed_tasks) if self.completed_tasks else 0.0,
            'average_battery_level': np.mean([user.battery_level for user in self.mobile_users]),
            'server_utilization': np.mean([server.current_load for server in self.edge_servers])
        }
        
        return metrics
    
    def reset_metrics(self):
        """Reset all performance tracking metrics."""
        self.completed_tasks.clear()
        self.dropped_tasks.clear()
        self.energy_consumption_history.clear()
        self.latency_history.clear()
        self.current_time = 0.0
        self.task_counter = 0
