"""
Gradient Sparsification Module
Implements Top-k gradient compression with error accumulation for federated learning
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class TopKSparsifier:
    """
    Top-k gradient sparsification with error accumulation.
    Selects the top-k% gradients by magnitude and accumulates pruned gradients.
    """
    
    def __init__(self, sparsity_ratio: float = 0.1, accumulate_error: bool = True):
        """
        Initialize the Top-k sparsifier.
        
        Args:
            sparsity_ratio: Fraction of gradients to keep (0.1 = 10%)
            accumulate_error: Whether to accumulate pruned gradients as error
        """
        self.sparsity_ratio = sparsity_ratio
        self.accumulate_error = accumulate_error
        self.error_accumulator = {}
        self.compression_stats = {
            'total_elements': 0,
            'sparse_elements': 0,
            'compression_ratio': 0.0
        }
    
    def sparsify_gradients(self, gradients: Dict[str, torch.Tensor], 
                          client_id: Optional[str] = None) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Apply Top-k sparsification to gradients.
        
        Args:
            gradients: Dictionary of parameter name -> gradient tensor
            client_id: Identifier for client (for error accumulation)
            
        Returns:
            Tuple of (sparse_gradients, sparsification_info)
        """
        sparse_gradients = {}
        sparsification_info = {
            'original_size': 0,
            'sparse_size': 0,
            'pruned_elements': 0,
            'compression_ratio': 0.0
        }
        
        # Add accumulated error if available
        if self.accumulate_error and client_id and client_id in self.error_accumulator:
            gradients = self._add_accumulated_error(gradients, client_id)
        
        for name, grad in gradients.items():
            if grad is None:
                sparse_gradients[name] = grad
                continue
                
            # Flatten gradient for top-k selection
            flat_grad = grad.view(-1)
            original_size = flat_grad.numel()
            
            # Calculate number of elements to keep
            k = max(1, int(original_size * self.sparsity_ratio))
            
            # Get top-k elements by absolute value
            _, indices = torch.topk(torch.abs(flat_grad), k)
            
            # Create sparse gradient
            sparse_flat = torch.zeros_like(flat_grad)
            sparse_flat[indices] = flat_grad[indices]
            sparse_grad = sparse_flat.view(grad.shape)
            
            sparse_gradients[name] = sparse_grad
            
            # Accumulate error (pruned gradients)
            if self.accumulate_error and client_id:
                error = grad - sparse_grad
                self._accumulate_error(name, error, client_id)
            
            # Update statistics
            sparsification_info['original_size'] += original_size
            sparsification_info['sparse_size'] += k
            sparsification_info['pruned_elements'] += (original_size - k)
        
        # Calculate compression ratio
        if sparsification_info['original_size'] > 0:
            sparsification_info['compression_ratio'] = (
                sparsification_info['sparse_size'] / sparsification_info['original_size']
            )
        
        self._update_global_stats(sparsification_info)
        
        return sparse_gradients, sparsification_info
    
    def _add_accumulated_error(self, gradients: Dict[str, torch.Tensor], 
                              client_id: str) -> Dict[str, torch.Tensor]:
        """Add accumulated error to current gradients."""
        if client_id not in self.error_accumulator:
            return gradients
            
        compensated_gradients = {}
        for name, grad in gradients.items():
            if grad is not None and name in self.error_accumulator[client_id]:
                error = self.error_accumulator[client_id][name]
                compensated_gradients[name] = grad + error
            else:
                compensated_gradients[name] = grad
                
        return compensated_gradients
    
    def _accumulate_error(self, param_name: str, error: torch.Tensor, client_id: str):
        """Accumulate error for future compensation."""
        if client_id not in self.error_accumulator:
            self.error_accumulator[client_id] = {}
            
        if param_name in self.error_accumulator[client_id]:
            self.error_accumulator[client_id][param_name] += error
        else:
            self.error_accumulator[client_id][param_name] = error.clone()
    
    def _update_global_stats(self, info: Dict):
        """Update global compression statistics."""
        self.compression_stats['total_elements'] += info['original_size']
        self.compression_stats['sparse_elements'] += info['sparse_size']
        
        if self.compression_stats['total_elements'] > 0:
            self.compression_stats['compression_ratio'] = (
                self.compression_stats['sparse_elements'] / 
                self.compression_stats['total_elements']
            )
    
    def get_compression_stats(self) -> Dict:
        """Get compression statistics."""
        return self.compression_stats.copy()
    
    def reset_error_accumulator(self, client_id: Optional[str] = None):
        """Reset error accumulator for specific client or all clients."""
        if client_id:
            if client_id in self.error_accumulator:
                del self.error_accumulator[client_id]
        else:
            self.error_accumulator.clear()
    
    def get_error_magnitude(self, client_id: str) -> float:
        """Get total magnitude of accumulated error for a client."""
        if client_id not in self.error_accumulator:
            return 0.0
            
        total_error = 0.0
        for param_name, error in self.error_accumulator[client_id].items():
            total_error += torch.norm(error).item()
            
        return total_error


class AdaptiveSparsifier:
    """
    Adaptive sparsification that adjusts compression based on network conditions.
    """
    
    def __init__(self, initial_sparsity: float = 0.1, 
                 min_sparsity: float = 0.05, max_sparsity: float = 0.5):
        """
        Initialize adaptive sparsifier.
        
        Args:
            initial_sparsity: Starting sparsity ratio
            min_sparsity: Minimum allowed sparsity
            max_sparsity: Maximum allowed sparsity
        """
        self.current_sparsity = initial_sparsity
        self.min_sparsity = min_sparsity
        self.max_sparsity = max_sparsity
        self.sparsifier = TopKSparsifier(initial_sparsity)
        self.adaptation_history = []
    
    def adapt_sparsity(self, network_condition: Dict) -> float:
        """
        Adapt sparsity based on network conditions.
        
        Args:
            network_condition: Dict with 'bandwidth', 'latency', 'loss_rate'
            
        Returns:
            New sparsity ratio
        """
        bandwidth = network_condition.get('bandwidth', 1.0)  # Mbps
        latency = network_condition.get('latency', 100)  # ms
        loss_rate = network_condition.get('loss_rate', 0.0)  # fraction
        
        # Simple adaptive strategy
        if bandwidth < 1.0 or latency > 200 or loss_rate > 0.1:
            # Poor network conditions - increase sparsity
            self.current_sparsity = min(self.max_sparsity, self.current_sparsity * 1.2)
        elif bandwidth > 5.0 and latency < 50 and loss_rate < 0.01:
            # Good network conditions - decrease sparsity
            self.current_sparsity = max(self.min_sparsity, self.current_sparsity * 0.9)
        
        self.sparsifier.sparsity_ratio = self.current_sparsity
        self.adaptation_history.append({
            'sparsity': self.current_sparsity,
            'network_condition': network_condition.copy()
        })
        
        logger.info(f"Adapted sparsity to {self.current_sparsity:.3f} based on network conditions")
        return self.current_sparsity
    
    def sparsify_gradients(self, gradients: Dict[str, torch.Tensor], 
                          client_id: Optional[str] = None) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """Sparsify gradients using current sparsity ratio."""
        return self.sparsifier.sparsify_gradients(gradients, client_id)
    
    def get_adaptation_history(self) -> List[Dict]:
        """Get history of sparsity adaptations."""
        return self.adaptation_history.copy()


class GradientCompressor:
    """
    Complete gradient compression system with multiple compression strategies.
    """
    
    def __init__(self, compression_config: Dict):
        """
        Initialize gradient compressor.
        
        Args:
            compression_config: Configuration for compression parameters
        """
        self.config = compression_config
        compression_type = compression_config.get('type', 'topk')
        sparsity_ratio = compression_config.get('sparsity_ratio', 0.1)
        
        if compression_type == 'topk':
            self.compressor = TopKSparsifier(sparsity_ratio)
        elif compression_type == 'adaptive':
            self.compressor = AdaptiveSparsifier(sparsity_ratio)
        else:
            raise ValueError(f"Unknown compression type: {compression_type}")
        
        self.compression_enabled = compression_config.get('enabled', True)
        
    def compress(self, gradients: Dict[str, torch.Tensor], 
                client_id: Optional[str] = None, 
                network_condition: Optional[Dict] = None) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Compress gradients using configured strategy.
        
        Args:
            gradients: Gradients to compress
            client_id: Client identifier
            network_condition: Current network conditions
            
        Returns:
            Compressed gradients and compression info
        """
        if not self.compression_enabled:
            info = {'compression_ratio': 1.0, 'original_size': 0, 'sparse_size': 0}
            return gradients, info
        
        # Adapt compression if using adaptive strategy
        if hasattr(self.compressor, 'adapt_sparsity') and network_condition:
            self.compressor.adapt_sparsity(network_condition)
        
        return self.compressor.sparsify_gradients(gradients, client_id)
    
    def get_stats(self) -> Dict:
        """Get compression statistics."""
        if hasattr(self.compressor, 'get_compression_stats'):
            return self.compressor.get_compression_stats()
        return {}
    
    def reset_client_state(self, client_id: str):
        """Reset state for specific client."""
        if hasattr(self.compressor, 'reset_error_accumulator'):
            self.compressor.reset_error_accumulator(client_id)
