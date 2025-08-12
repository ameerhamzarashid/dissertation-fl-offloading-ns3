"""
Core FL Framework - Main Coordination Logic
Handles the overall federated learning orchestration between clients and server
"""

import logging
import threading
import time
from typing import Dict, List, Any, Optional
from collections import defaultdict

import torch
import numpy as np

from agents.dueling_dqn import DuelingDQN
from agents.prioritized_replay import PrioritizedReplayBuffer
from simulation.mec_environment import MECEnvironment
from utils.logger import setup_logger
from utils.file_manager import save_model, load_model


class FLCoordinator:
    """
    Main coordinator for federated learning operations.
    Manages communication between clients and server, handles model distribution and aggregation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the FL coordinator with configuration parameters.
        
        Args:
            config: Dictionary containing all system configuration parameters
        """
        self.config = config
        self.logger = setup_logger("FLCoordinator")
        
        # Network and learning parameters
        self.num_clients = config['network']['num_mobile_users']
        self.num_servers = config['network']['num_edge_servers']
        self.global_rounds = config['learning']['global_rounds']
        self.local_epochs = config['learning']['local_epochs']
        
        # Model and environment setup
        self.state_dim = config['learning']['state_dim']
        self.action_dim = config['learning']['action_dim']
        self.hidden_dim = config['learning']['hidden_dim']
        
        # Initialize global model
        self.global_model = DuelingDQN(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim
        )
        
        # Client management
        self.active_clients = set()
        self.client_models = {}
        self.client_updates = {}
        self.client_weights = {}
        
        # Metrics tracking
        self.round_metrics = defaultdict(list)
        self.communication_costs = []
        self.convergence_history = []
        
        # Threading for concurrent operations
        self.lock = threading.Lock()
        self.round_complete = threading.Event()
        
        self.logger.info(f"FL Coordinator initialized with {self.num_clients} clients and {self.num_servers} edge servers")
    
    def initialize_clients(self):
        """
        Initialize all client models with the global model parameters.
        """
        self.logger.info("Initializing client models")
        
        global_state_dict = self.global_model.state_dict()
        
        for client_id in range(self.num_clients):
            # Create local model copy for each client
            client_model = DuelingDQN(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=self.hidden_dim
            )
            client_model.load_state_dict(global_state_dict.copy())
            
            self.client_models[client_id] = client_model
            self.active_clients.add(client_id)
            
            # Initialize client weight based on expected data contribution
            self.client_weights[client_id] = 1.0 / self.num_clients
        
        self.logger.info(f"Initialized {len(self.client_models)} client models")
    
    def distribute_global_model(self) -> Dict[int, torch.nn.Module]:
        """
        Distribute the current global model to all active clients.
        
        Returns:
            Dictionary mapping client IDs to model copies
        """
        with self.lock:
            global_state_dict = self.global_model.state_dict()
            
            for client_id in self.active_clients:
                self.client_models[client_id].load_state_dict(global_state_dict.copy())
            
            self.logger.debug(f"Distributed global model to {len(self.active_clients)} clients")
            return self.client_models.copy()
    
    def collect_client_updates(self, client_updates: Dict[int, Dict[str, torch.Tensor]]):
        """
        Collect model updates from clients after local training.
        
        Args:
            client_updates: Dictionary mapping client IDs to their model parameter updates
        """
        with self.lock:
            self.client_updates = client_updates.copy()
            
            # Calculate communication cost for this round
            total_params = 0
            for client_id, update in client_updates.items():
                client_params = sum(param.numel() for param in update.values())
                total_params += client_params
            
            # Assuming 32-bit floats (4 bytes per parameter)
            communication_cost = total_params * 4
            self.communication_costs.append(communication_cost)
            
            self.logger.info(f"Collected updates from {len(client_updates)} clients")
            self.logger.info(f"Communication cost this round: {communication_cost / (1024*1024):.2f} MB")
    
    def aggregate_models(self) -> torch.nn.Module:
        """
        Perform federated averaging on collected client updates.
        
        Returns:
            Updated global model
        """
        with self.lock:
            if not self.client_updates:
                self.logger.warning("No client updates available for aggregation")
                return self.global_model
            
            # Initialize aggregated parameters
            global_state_dict = self.global_model.state_dict()
            aggregated_state_dict = {}
            
            # Initialize with zeros
            for key in global_state_dict.keys():
                aggregated_state_dict[key] = torch.zeros_like(global_state_dict[key])
            
            # Weighted aggregation of client updates
            total_weight = 0.0
            for client_id, update in self.client_updates.items():
                client_weight = self.client_weights.get(client_id, 1.0)
                total_weight += client_weight
                
                for key, param in update.items():
                    aggregated_state_dict[key] += client_weight * param
            
            # Normalize by total weight
            for key in aggregated_state_dict.keys():
                aggregated_state_dict[key] /= total_weight
            
            # Update global model
            self.global_model.load_state_dict(aggregated_state_dict)
            
            self.logger.info(f"Aggregated updates from {len(self.client_updates)} clients")
            return self.global_model
    
    def evaluate_global_model(self, test_environment: MECEnvironment) -> Dict[str, float]:
        """
        Evaluate the current global model performance.
        
        Args:
            test_environment: MEC environment for testing
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.global_model.eval()
        total_reward = 0.0
        total_steps = 0
        episode_rewards = []
        
        # Run evaluation episodes
        num_eval_episodes = self.config.get('evaluation', {}).get('num_episodes', 10)
        
        for episode in range(num_eval_episodes):
            state = test_environment.reset()
            episode_reward = 0.0
            done = False
            
            while not done and total_steps < 1000:  # Max steps per episode
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action = self.global_model.act(state_tensor, epsilon=0.0)  # Greedy action
                
                next_state, reward, done = test_environment.step(action)
                episode_reward += reward
                total_reward += reward
                total_steps += 1
                
                state = next_state
            
            episode_rewards.append(episode_reward)
        
        # Calculate metrics
        metrics = {
            'average_reward': total_reward / num_eval_episodes,
            'average_episode_length': total_steps / num_eval_episodes,
            'reward_std': np.std(episode_rewards),
            'min_reward': min(episode_rewards),
            'max_reward': max(episode_rewards)
        }
        
        self.convergence_history.append(metrics['average_reward'])
        self.global_model.train()
        
        return metrics
    
    def update_client_weights(self, client_performance: Dict[int, float]):
        """
        Update client weights based on their contribution quality.
        
        Args:
            client_performance: Dictionary mapping client IDs to performance scores
        """
        with self.lock:
            if not client_performance:
                return
            
            # Normalize performance scores to weights
            total_performance = sum(client_performance.values())
            if total_performance > 0:
                for client_id, performance in client_performance.items():
                    self.client_weights[client_id] = performance / total_performance
            
            self.logger.debug("Updated client weights based on performance")
    
    def run_federated_round(self, round_num: int, mec_environment: MECEnvironment) -> Dict[str, Any]:
        """
        Execute a complete federated learning round.
        
        Args:
            round_num: Current round number
            mec_environment: MEC simulation environment
            
        Returns:
            Round metrics and results
        """
        self.logger.info(f"Starting federated round {round_num}")
        round_start_time = time.time()
        
        # Step 1: Distribute global model to clients
        client_models = self.distribute_global_model()
        
        # Step 2: Simulate local training (this would be done by actual clients)
        client_updates = {}
        client_performance = {}
        
        for client_id in self.active_clients:
            # Simulate local training for this client
            local_update, performance = self._simulate_local_training(
                client_id, client_models[client_id], mec_environment
            )
            client_updates[client_id] = local_update
            client_performance[client_id] = performance
        
        # Step 3: Collect and aggregate updates
        self.collect_client_updates(client_updates)
        self.aggregate_models()
        
        # Step 4: Update client weights based on performance
        self.update_client_weights(client_performance)
        
        # Step 5: Evaluate global model
        eval_metrics = self.evaluate_global_model(mec_environment)
        
        # Calculate round metrics
        round_time = time.time() - round_start_time
        round_metrics = {
            'round': round_num,
            'round_time': round_time,
            'communication_cost': self.communication_costs[-1] if self.communication_costs else 0,
            'num_participants': len(client_updates),
            **eval_metrics
        }
        
        self.round_metrics['round_data'].append(round_metrics)
        
        self.logger.info(f"Round {round_num} completed in {round_time:.2f}s, "
                        f"avg reward: {eval_metrics['average_reward']:.4f}")
        
        return round_metrics
    
    def _simulate_local_training(self, client_id: int, model: torch.nn.Module, 
                                environment: MECEnvironment) -> tuple:
        """
        Simulate local training for a client.
        
        Args:
            client_id: Client identifier
            model: Local model to train
            environment: MEC environment for interaction
            
        Returns:
            Tuple of (model_update, performance_score)
        """
        # Store initial model state
        initial_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        
        # Simulate local training episodes
        total_reward = 0.0
        episodes_completed = 0
        
        for epoch in range(self.local_epochs):
            state = environment.reset()
            episode_reward = 0.0
            done = False
            steps = 0
            
            while not done and steps < 100:  # Max steps per episode
                # Select action using current model
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action = model.act(state_tensor, epsilon=0.1)  # Exploration
                
                next_state, reward, done = environment.step(action)
                episode_reward += reward
                steps += 1
                
                state = next_state
            
            total_reward += episode_reward
            episodes_completed += 1
        
        # Calculate model update (difference from initial)
        final_state_dict = model.state_dict()
        model_update = {}
        for key in initial_state_dict.keys():
            model_update[key] = final_state_dict[key] - initial_state_dict[key]
        
        # Performance score for this client
        performance_score = total_reward / max(episodes_completed, 1)
        
        return model_update, performance_score
    
    def save_checkpoint(self, round_num: int, save_path: str):
        """
        Save current state as checkpoint.
        
        Args:
            round_num: Current round number
            save_path: Path to save checkpoint
        """
        checkpoint = {
            'round': round_num,
            'global_model_state': self.global_model.state_dict(),
            'client_weights': self.client_weights,
            'communication_costs': self.communication_costs,
            'convergence_history': self.convergence_history,
            'round_metrics': dict(self.round_metrics)
        }
        
        save_model(checkpoint, f"{save_path}/checkpoint_round_{round_num}.pt")
        self.logger.info(f"Saved checkpoint for round {round_num}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get training summary.
        
        Returns:
            Dictionary containing training statistics and metrics
        """
        if not self.communication_costs or not self.convergence_history:
            return {"status": "No training data available"}
        
        total_communication = sum(self.communication_costs)
        avg_communication_per_round = total_communication / len(self.communication_costs)
        
        summary = {
            'total_rounds': len(self.communication_costs),
            'total_communication_mb': total_communication / (1024 * 1024),
            'avg_communication_per_round_mb': avg_communication_per_round / (1024 * 1024),
            'final_performance': self.convergence_history[-1],
            'best_performance': max(self.convergence_history),
            'performance_improvement': self.convergence_history[-1] - self.convergence_history[0],
            'num_active_clients': len(self.active_clients),
            'convergence_trend': self.convergence_history[-10:] if len(self.convergence_history) >= 10 else self.convergence_history
        }
        
        return summary
