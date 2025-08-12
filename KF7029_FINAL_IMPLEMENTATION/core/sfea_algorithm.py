"""
Enhanced Federated Learning Algorithm with Gradient Sparsification
Implements SFEA (Sparsification-based Federated Edge-AI) algorithm
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import logging
import copy
import numpy as np

from .gradient_sparsification import GradientCompressor
from .fl_coordinator import FLCoordinator

logger = logging.getLogger(__name__)


class SFEAFederatedLearning:
    """
    Sparsification-based Federated Edge-AI (SFEA) Algorithm.
    Enhanced federated learning with gradient sparsification and communication optimization.
    """
    
    def __init__(self, config: Dict, model: nn.Module):
        """
        Initialize SFEA federated learning system.
        
        Args:
            config: Configuration dictionary
            model: Global model to be trained
        """
        self.config = config
        self.global_model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_model.to(self.device)
        
        # Initialize gradient compressor
        compression_config = config.get('compression', {
            'type': 'topk',
            'sparsity_ratio': 0.1,
            'enabled': True
        })
        self.gradient_compressor = GradientCompressor(compression_config)
        
        # Federated learning parameters
        self.num_clients = config['network']['num_mobile_users']
        self.num_rounds = config['learning']['global_rounds']
        self.client_fraction = config['learning'].get('client_fraction', 1.0)
        self.local_epochs = config['learning']['local_epochs']
        
        # SFEA specific parameters
        self.convergence_threshold = config['learning'].get('convergence_threshold', 1e-4)
        self.communication_budget = config.get('communication_budget', float('inf'))
        self.used_communication = 0
        
        # Training state
        self.current_round = 0
        self.training_history = []
        self.convergence_metrics = []
        self.communication_costs = []
        
        logger.info(f"Initialized SFEA with {self.num_clients} clients, "
                   f"sparsity ratio: {compression_config.get('sparsity_ratio', 0.1)}")
    
    def run_federated_training(self, client_datasets: Dict, 
                             validation_data: Optional[Any] = None) -> Dict:
        """
        Run complete SFEA federated training.
        
        Args:
            client_datasets: Dictionary mapping client_id -> dataset
            validation_data: Optional validation dataset
            
        Returns:
            Training results and statistics
        """
        logger.info("Starting SFEA federated training")
        
        for round_num in range(self.num_rounds):
            self.current_round = round_num
            
            # Check communication budget
            if self.used_communication >= self.communication_budget:
                logger.warning(f"Communication budget exhausted at round {round_num}")
                break
            
            # Execute federated round
            round_results = self._execute_federated_round(client_datasets, validation_data)
            
            # Check convergence
            if self._check_convergence(round_results):
                logger.info(f"Converged at round {round_num}")
                break
            
            # Log round results
            self.training_history.append(round_results)
            logger.info(f"Round {round_num}: {round_results['summary']}")
        
        # Compile final results
        final_results = self._compile_training_results()
        logger.info("SFEA federated training completed")
        
        return final_results
    
    def _execute_federated_round(self, client_datasets: Dict, 
                                validation_data: Optional[Any] = None) -> Dict:
        """Execute single federated learning round with SFEA."""
        round_start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        round_end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if round_start_time:
            round_start_time.record()
        
        # 1. Client selection
        selected_clients = self._select_clients()
        
        # 2. Distribute global model
        client_models = self._distribute_global_model(selected_clients)
        
        # 3. Local training with sparsification
        client_updates, update_info = self._perform_local_training(
            selected_clients, client_datasets, client_models
        )
        
        # 4. Aggregate sparse updates
        aggregation_results = self._aggregate_sparse_updates(client_updates, update_info)
        
        # 5. Update global model
        self._update_global_model(aggregation_results['aggregated_gradients'])
        
        # 6. Evaluate if validation data available
        validation_results = {}
        if validation_data:
            validation_results = self._evaluate_global_model(validation_data)
        
        if round_end_time:
            round_end_time.record()
            torch.cuda.synchronize()
            round_time = round_start_time.elapsed_time(round_end_time)
        else:
            round_time = 0
        
        # Compile round results
        round_results = {
            'round': self.current_round,
            'selected_clients': len(selected_clients),
            'communication_cost': update_info['total_communication_cost'],
            'compression_ratio': update_info['avg_compression_ratio'],
            'aggregation_quality': aggregation_results['quality_metrics'],
            'validation_results': validation_results,
            'round_time': round_time,
            'summary': {
                'comm_cost': update_info['total_communication_cost'],
                'compression': f"{update_info['avg_compression_ratio']:.3f}",
                'clients': len(selected_clients)
            }
        }
        
        # Update communication usage
        self.used_communication += update_info['total_communication_cost']
        self.communication_costs.append(update_info['total_communication_cost'])
        
        return round_results
    
    def _select_clients(self) -> List[int]:
        """Select clients for current round."""
        num_selected = max(1, int(self.num_clients * self.client_fraction))
        return np.random.choice(self.num_clients, num_selected, replace=False).tolist()
    
    def _distribute_global_model(self, selected_clients: List[int]) -> Dict[int, nn.Module]:
        """Distribute global model to selected clients."""
        client_models = {}
        for client_id in selected_clients:
            client_models[client_id] = copy.deepcopy(self.global_model)
        return client_models
    
    def _perform_local_training(self, selected_clients: List[int], 
                               client_datasets: Dict, 
                               client_models: Dict[int, nn.Module]) -> Tuple[Dict, Dict]:
        """Perform local training on selected clients with sparsification."""
        client_updates = {}
        compression_stats = []
        total_communication_cost = 0
        
        for client_id in selected_clients:
            # Get client data
            client_data = client_datasets.get(client_id)
            if client_data is None:
                logger.warning(f"No data for client {client_id}")
                continue
            
            # Local training
            client_model = client_models[client_id]
            local_gradients = self._train_client_model(client_model, client_data)
            
            # Network condition simulation (could be real network monitoring)
            network_condition = self._simulate_network_condition(client_id)
            
            # Apply gradient sparsification
            sparse_gradients, compression_info = self.gradient_compressor.compress(
                local_gradients, str(client_id), network_condition
            )
            
            client_updates[client_id] = sparse_gradients
            compression_stats.append(compression_info)
            
            # Calculate communication cost (proportional to transmitted data)
            comm_cost = compression_info.get('sparse_size', 0)
            total_communication_cost += comm_cost
        
        # Aggregate compression statistics
        if compression_stats:
            avg_compression_ratio = np.mean([s.get('compression_ratio', 1.0) for s in compression_stats])
            total_pruned = sum(s.get('pruned_elements', 0) for s in compression_stats)
        else:
            avg_compression_ratio = 1.0
            total_pruned = 0
        
        update_info = {
            'total_communication_cost': total_communication_cost,
            'avg_compression_ratio': avg_compression_ratio,
            'total_pruned_elements': total_pruned,
            'compression_stats': compression_stats
        }
        
        return client_updates, update_info
    
    def _train_client_model(self, model: nn.Module, client_data: Any) -> Dict[str, torch.Tensor]:
        """Train client model locally and return gradients."""
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), 
                                   lr=self.config['learning']['learning_rate'])
        
        # Store initial parameters
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Local training loop
        for epoch in range(self.local_epochs):
            for batch in client_data:  # Assuming client_data is iterable
                optimizer.zero_grad()
                
                # Forward pass (implementation depends on your model/data)
                # This is a placeholder - implement according to your specific model
                loss = self._compute_loss(model, batch)
                
                # Backward pass
                loss.backward()
                optimizer.step()
        
        # Calculate gradients as difference from initial parameters
        gradients = {}
        for name, param in model.named_parameters():
            gradients[name] = initial_params[name] - param
        
        return gradients
    
    def _compute_loss(self, model: nn.Module, batch: Any) -> torch.Tensor:
        """Compute loss for a batch. Override this method for your specific model."""
        # Placeholder implementation
        # In practice, this should implement your specific loss computation
        return torch.tensor(0.0, requires_grad=True)
    
    def _simulate_network_condition(self, client_id: int) -> Dict:
        """Simulate network conditions. Replace with real network monitoring."""
        # Simulate varying network conditions
        base_bandwidth = 5.0  # Mbps
        base_latency = 50  # ms
        base_loss_rate = 0.01
        
        # Add some randomness
        bandwidth = base_bandwidth * (0.5 + np.random.random())
        latency = base_latency * (0.5 + np.random.random() * 2)
        loss_rate = base_loss_rate * np.random.random()
        
        return {
            'bandwidth': bandwidth,
            'latency': latency,
            'loss_rate': loss_rate
        }
    
    def _aggregate_sparse_updates(self, client_updates: Dict, update_info: Dict) -> Dict:
        """Aggregate sparse client updates using weighted averaging."""
        if not client_updates:
            return {'aggregated_gradients': {}, 'quality_metrics': {}}
        
        # Initialize aggregated gradients
        aggregated_gradients = {}
        param_names = list(next(iter(client_updates.values())).keys())
        
        for param_name in param_names:
            # Collect all client gradients for this parameter
            client_grads = []
            for client_id, gradients in client_updates.items():
                if param_name in gradients and gradients[param_name] is not None:
                    client_grads.append(gradients[param_name])
            
            if client_grads:
                # Simple average aggregation (can be enhanced with weights)
                aggregated_gradients[param_name] = torch.stack(client_grads).mean(dim=0)
            else:
                aggregated_gradients[param_name] = None
        
        # Calculate quality metrics
        quality_metrics = {
            'num_participating_clients': len(client_updates),
            'avg_sparsity': update_info.get('avg_compression_ratio', 1.0),
            'total_communication_cost': update_info.get('total_communication_cost', 0)
        }
        
        return {
            'aggregated_gradients': aggregated_gradients,
            'quality_metrics': quality_metrics
        }
    
    def _update_global_model(self, aggregated_gradients: Dict[str, torch.Tensor]):
        """Update global model with aggregated gradients."""
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in aggregated_gradients and aggregated_gradients[name] is not None:
                    param.data -= aggregated_gradients[name]
    
    def _evaluate_global_model(self, validation_data: Any) -> Dict:
        """Evaluate global model on validation data."""
        self.global_model.eval()
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in validation_data:
                loss = self._compute_loss(self.global_model, batch)
                total_loss += loss.item()
                num_samples += 1
        
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'num_samples': num_samples
        }
    
    def _check_convergence(self, round_results: Dict) -> bool:
        """Check if training has converged."""
        if not self.convergence_metrics:
            self.convergence_metrics.append(round_results)
            return False
        
        # Simple convergence check based on loss change
        current_loss = round_results.get('validation_results', {}).get('loss', float('inf'))
        previous_loss = self.convergence_metrics[-1].get('validation_results', {}).get('loss', float('inf'))
        
        if abs(current_loss - previous_loss) < self.convergence_threshold:
            return True
        
        self.convergence_metrics.append(round_results)
        return False
    
    def _compile_training_results(self) -> Dict:
        """Compile final training results."""
        total_communication_cost = sum(self.communication_costs)
        avg_compression_ratio = np.mean([
            h['compression_ratio'] for h in self.training_history
        ]) if self.training_history else 1.0
        
        return {
            'total_rounds': len(self.training_history),
            'total_communication_cost': total_communication_cost,
            'average_compression_ratio': avg_compression_ratio,
            'convergence_achieved': len(self.convergence_metrics) > 0,
            'final_model_state': self.global_model.state_dict(),
            'training_history': self.training_history,
            'communication_efficiency': {
                'total_cost': total_communication_cost,
                'cost_per_round': total_communication_cost / max(1, len(self.training_history)),
                'compression_savings': 1 - avg_compression_ratio
            }
        }
    
    def get_global_model(self) -> nn.Module:
        """Get current global model."""
        return self.global_model
    
    def get_compression_stats(self) -> Dict:
        """Get gradient compression statistics."""
        return self.gradient_compressor.get_stats()
    
    def reset_client_states(self):
        """Reset all client states (e.g., error accumulators)."""
        for client_id in range(self.num_clients):
            self.gradient_compressor.reset_client_state(str(client_id))
