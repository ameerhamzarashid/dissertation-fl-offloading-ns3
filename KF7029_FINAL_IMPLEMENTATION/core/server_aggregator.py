"""
Server Aggregator - Handles model aggregation and federated averaging
Implements the core federated learning aggregation algorithms
"""

import time
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np

from utils.logger import setup_logger


class ServerAggregator:
    """
    Handles federated model aggregation using various aggregation strategies.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the server aggregator.
        
        Args:
            config: System configuration dictionary
        """
        self.config = config
        self.logger = setup_logger("ServerAggregator")
        
        # Aggregation parameters
        self.aggregation_method = config.get('aggregation', {}).get('method', 'fedavg')
        self.min_clients_per_round = config.get('aggregation', {}).get('min_clients', 1)
        self.client_selection_strategy = config.get('aggregation', {}).get('client_selection', 'all')
        
        # Model versioning and tracking
        self.global_model_version = 0
        self.aggregation_history = []
        self.client_contributions = defaultdict(list)
        
        # Performance tracking
        self.aggregation_times = []
        self.model_divergence_history = []
        
        self.logger.info(f"Server Aggregator initialized with method: {self.aggregation_method}")
    
    def federated_averaging(self, 
                           client_updates: Dict[str, Dict[str, torch.Tensor]], 
                           client_weights: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """
        Perform standard federated averaging on client model updates.
        
        Args:
            client_updates: Dictionary mapping client IDs to their model parameter updates
            client_weights: Optional weights for each client (defaults to equal weights)
            
        Returns:
            Aggregated model parameters
        """
        start_time = time.time()
        
        if not client_updates:
            self.logger.warning("No client updates provided for aggregation")
            return {}
        
        # Set default equal weights if not provided
        if client_weights is None:
            num_clients = len(client_updates)
            client_weights = {client_id: 1.0 / num_clients for client_id in client_updates.keys()}
        
        # Normalize weights to sum to 1
        total_weight = sum(client_weights.values())
        if total_weight > 0:
            client_weights = {k: v / total_weight for k, v in client_weights.items()}
        
        # Initialize aggregated parameters
        aggregated_params = {}
        
        # Get parameter names from first client
        first_client_id = next(iter(client_updates.keys()))
        param_names = list(client_updates[first_client_id].keys())
        
        # Initialize aggregated parameters with zeros
        for param_name in param_names:
            param_shape = client_updates[first_client_id][param_name].shape
            aggregated_params[param_name] = torch.zeros(param_shape, dtype=torch.float32)
        
        # Weighted aggregation
        for client_id, client_update in client_updates.items():
            client_weight = client_weights.get(client_id, 0.0)
            
            for param_name, param_tensor in client_update.items():
                aggregated_params[param_name] += client_weight * param_tensor
        
        # Record aggregation statistics
        aggregation_time = time.time() - start_time
        self.aggregation_times.append(aggregation_time)
        
        self.logger.info(f"Federated averaging completed for {len(client_updates)} clients "
                        f"in {aggregation_time:.4f}s")
        
        # Update tracking
        self.global_model_version += 1
        self._record_aggregation(client_updates, client_weights, aggregation_time)
        
        return aggregated_params
    
    def weighted_federated_averaging(self, 
                                   client_updates: Dict[str, Dict[str, torch.Tensor]], 
                                   client_data_sizes: Dict[str, int]) -> Dict[str, torch.Tensor]:
        """
        Perform federated averaging weighted by client data sizes.
        
        Args:
            client_updates: Dictionary mapping client IDs to their model parameter updates
            client_data_sizes: Dictionary mapping client IDs to their local data sizes
            
        Returns:
            Aggregated model parameters
        """
        # Calculate weights based on data sizes
        total_data_size = sum(client_data_sizes.values())
        client_weights = {}
        
        for client_id in client_updates.keys():
            data_size = client_data_sizes.get(client_id, 1)
            client_weights[client_id] = data_size / total_data_size
        
        return self.federated_averaging(client_updates, client_weights)
    
    def performance_weighted_averaging(self, 
                                     client_updates: Dict[str, Dict[str, torch.Tensor]], 
                                     client_performance: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """
        Perform federated averaging weighted by client performance scores.
        
        Args:
            client_updates: Dictionary mapping client IDs to their model parameter updates
            client_performance: Dictionary mapping client IDs to performance scores
            
        Returns:
            Aggregated model parameters
        """
        # Handle negative performance scores by shifting to positive range
        min_performance = min(client_performance.values())
        if min_performance < 0:
            shifted_performance = {k: v - min_performance + 1e-6 
                                 for k, v in client_performance.items()}
        else:
            shifted_performance = client_performance
        
        # Calculate weights based on performance
        total_performance = sum(shifted_performance.values())
        client_weights = {}
        
        for client_id in client_updates.keys():
            performance = shifted_performance.get(client_id, 1e-6)
            client_weights[client_id] = performance / total_performance
        
        return self.federated_averaging(client_updates, client_weights)
    
    def stable_aggregation(self, 
                          client_updates: Dict[str, Dict[str, torch.Tensor]], 
                          trim_ratio: float = 0.1) -> Dict[str, torch.Tensor]:
        """
        Perform stable aggregation by trimming extreme updates.
        
        Args:
            client_updates: Dictionary mapping client IDs to their model parameter updates
            trim_ratio: Fraction of extreme values to trim (0.1 = trim 10% from each end)
            
        Returns:
            Aggregated model parameters
        """
        if not client_updates:
            return {}
        
        num_clients = len(client_updates)
        trim_count = max(1, int(num_clients * trim_ratio))
        
        # Get parameter names
        first_client_id = next(iter(client_updates.keys()))
        param_names = list(client_updates[first_client_id].keys())
        
        aggregated_params = {}
        
        for param_name in param_names:
            # Collect all client values for this parameter
            param_values = []
            client_ids = []
            
            for client_id, client_update in client_updates.items():
                param_values.append(client_update[param_name])
                client_ids.append(client_id)
            
            # Stack all parameter tensors
            stacked_params = torch.stack(param_values, dim=0)
            
            # Calculate element-wise median stable center
            median_params, _ = torch.median(stacked_params, dim=0)
            
            # Calculate distances from median
            distances = torch.norm(stacked_params - median_params.unsqueeze(0), dim=tuple(range(1, stacked_params.ndim)))
            
            # Select clients excluding the most extreme ones
            if num_clients > 2 * trim_count:
                _, sorted_indices = torch.sort(distances)
                selected_indices = sorted_indices[trim_count:-trim_count] if trim_count > 0 else sorted_indices
                selected_params = stacked_params[selected_indices]
                
                # Average the selected parameters
                aggregated_params[param_name] = torch.mean(selected_params, dim=0)
            else:
                # If too few clients, just use median
                aggregated_params[param_name] = median_params
        
        self.logger.info(f"Stable aggregation completed, used {num_clients - 2 * trim_count} out of {num_clients} clients")
        
        return aggregated_params
    
    def aggregate(self, 
                  client_updates: Dict[str, Dict[str, torch.Tensor]], 
                  aggregation_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        Main aggregation method that dispatches to specific aggregation strategies.
        
        Args:
            client_updates: Dictionary mapping client IDs to their model parameter updates
            aggregation_metadata: Additional metadata for aggregation (weights, performance, etc.)
            
        Returns:
            Aggregated model parameters
        """
        if len(client_updates) < self.min_clients_per_round:
            self.logger.warning(f"Insufficient clients for aggregation: {len(client_updates)} < {self.min_clients_per_round}")
            return {}
        
        if not aggregation_metadata:
            aggregation_metadata = {}
        
        # Select aggregation method
        if self.aggregation_method == 'fedavg':
            client_weights = aggregation_metadata.get('client_weights')
            return self.federated_averaging(client_updates, client_weights)
        
        elif self.aggregation_method == 'weighted_fedavg':
            client_data_sizes = aggregation_metadata.get('client_data_sizes', {})
            return self.weighted_federated_averaging(client_updates, client_data_sizes)
        
        elif self.aggregation_method == 'performance_weighted':
            client_performance = aggregation_metadata.get('client_performance', {})
            return self.performance_weighted_averaging(client_updates, client_performance)
        
        elif self.aggregation_method == 'stable':
            trim_ratio = aggregation_metadata.get('trim_ratio', 0.1)
            return self.stable_aggregation(client_updates, trim_ratio)
        
        else:
            self.logger.error(f"Unknown aggregation method: {self.aggregation_method}")
            return self.federated_averaging(client_updates)
    
    def calculate_model_divergence(self, 
                                  client_updates: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """
        Calculate divergence metrics between client updates.
        
        Args:
            client_updates: Dictionary mapping client IDs to their model parameter updates
            
        Returns:
            Dictionary containing divergence metrics
        """
        if len(client_updates) < 2:
            return {'average_divergence': 0.0, 'max_divergence': 0.0, 'min_divergence': 0.0}
        
        client_ids = list(client_updates.keys())
        divergences = []
        
        # Calculate pairwise divergences
        for i in range(len(client_ids)):
            for j in range(i + 1, len(client_ids)):
                client1_update = client_updates[client_ids[i]]
                client2_update = client_updates[client_ids[j]]
                
                # Calculate L2 distance between updates
                total_divergence = 0.0
                total_params = 0
                
                for param_name in client1_update.keys():
                    if param_name in client2_update:
                        param1 = client1_update[param_name]
                        param2 = client2_update[param_name]
                        
                        divergence = torch.norm(param1 - param2).item()
                        total_divergence += divergence
                        total_params += param1.numel()
                
                if total_params > 0:
                    avg_divergence = total_divergence / total_params
                    divergences.append(avg_divergence)
        
        if divergences:
            avg_divergence = np.mean(divergences)
            max_divergence = np.max(divergences)
            min_divergence = np.min(divergences)
        else:
            avg_divergence = max_divergence = min_divergence = 0.0
        
        divergence_metrics = {
            'average_divergence': avg_divergence,
            'max_divergence': max_divergence,
            'min_divergence': min_divergence,
            'std_divergence': np.std(divergences) if divergences else 0.0
        }
        
        self.model_divergence_history.append(divergence_metrics)
        
        return divergence_metrics
    
    def _record_aggregation(self, 
                           client_updates: Dict[str, Dict[str, torch.Tensor]], 
                           client_weights: Dict[str, float], 
                           aggregation_time: float):
        """
        Record aggregation statistics for analysis.
        
        Args:
            client_updates: Client model updates that were aggregated
            client_weights: Weights used for aggregation
            aggregation_time: Time taken for aggregation
        """
        # Calculate divergence metrics
        divergence_metrics = self.calculate_model_divergence(client_updates)
        
        # Record aggregation round info
        aggregation_record = {
            'version': self.global_model_version,
            'timestamp': time.time(),
            'num_clients': len(client_updates),
            'client_ids': list(client_updates.keys()),
            'client_weights': client_weights.copy(),
            'aggregation_time': aggregation_time,
            'divergence_metrics': divergence_metrics
        }
        
        self.aggregation_history.append(aggregation_record)
        
        # Update client contribution tracking
        for client_id in client_updates.keys():
            self.client_contributions[client_id].append({
                'version': self.global_model_version,
                'weight': client_weights.get(client_id, 0.0),
                'timestamp': time.time()
            })
    
    def get_aggregation_statistics(self) -> Dict[str, Any]:
        """
        Get aggregation statistics.
        
        Returns:
            Dictionary containing aggregation statistics
        """
        if not self.aggregation_history:
            return {'status': 'No aggregation history available'}
        
        # Calculate statistics
        total_rounds = len(self.aggregation_history)
        avg_clients_per_round = np.mean([record['num_clients'] for record in self.aggregation_history])
        avg_aggregation_time = np.mean(self.aggregation_times)
        
        # Divergence trends
        recent_divergences = [record['divergence_metrics']['average_divergence'] 
                            for record in self.aggregation_history[-10:]]
        avg_recent_divergence = np.mean(recent_divergences) if recent_divergences else 0.0
        
        # Client participation statistics
        all_clients = set()
        for record in self.aggregation_history:
            all_clients.update(record['client_ids'])
        
        client_participation = {}
        for client_id in all_clients:
            participation_count = sum(1 for record in self.aggregation_history 
                                    if client_id in record['client_ids'])
            client_participation[client_id] = participation_count / total_rounds
        
        statistics = {
            'total_aggregation_rounds': total_rounds,
            'current_model_version': self.global_model_version,
            'average_clients_per_round': avg_clients_per_round,
            'average_aggregation_time': avg_aggregation_time,
            'total_unique_clients': len(all_clients),
            'average_recent_divergence': avg_recent_divergence,
            'client_participation_rates': client_participation,
            'aggregation_method': self.aggregation_method
        }
        
        return statistics
    
    def get_client_contribution_summary(self, client_id: str) -> Dict[str, Any]:
        """
        Get contribution summary for a specific client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Dictionary containing client contribution statistics
        """
        contributions = self.client_contributions.get(client_id, [])
        
        if not contributions:
            return {'status': 'No contributions found for this client'}
        
        total_contributions = len(contributions)
        total_weight = sum(contrib['weight'] for contrib in contributions)
        avg_weight = total_weight / total_contributions
        
        # Recent contribution trend
        recent_contributions = contributions[-5:] if len(contributions) >= 5 else contributions
        recent_avg_weight = sum(contrib['weight'] for contrib in recent_contributions) / len(recent_contributions)
        
        summary = {
            'client_id': client_id,
            'total_contributions': total_contributions,
            'total_aggregated_weight': total_weight,
            'average_weight_per_round': avg_weight,
            'recent_average_weight': recent_avg_weight,
            'first_contribution': contributions[0]['timestamp'],
            'last_contribution': contributions[-1]['timestamp'],
            'participation_rate': total_contributions / len(self.aggregation_history) if self.aggregation_history else 0.0
        }
        
        return summary
