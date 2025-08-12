"""
Synthetic Data Generator for Federated Learning
Generates realistic datasets for training and testing SFEA implementation
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import sys
import os

# Add current directory to path for GPU utilities
try:
    from gpu_utils import setup_device, optimize_model_for_gpu
except ImportError:
    # Fallback for basic device setup
    def setup_device(preferred_device=None):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def optimize_model_for_gpu(model, device):
        return model.to(device)

logger = logging.getLogger(__name__)


class MobileTaskDataset(Dataset):
    """
    Synthetic dataset for mobile edge computing task classification.
    Generates features representing mobile device tasks and their optimal offloading decisions.
    """
    
    def __init__(self, num_samples: int = 1000, num_features: int = 12, num_classes: int = 6):
        """
        Initialize synthetic mobile task dataset.
        
        Args:
            num_samples: Number of samples to generate
            num_features: Number of input features (matching MEC environment state)
            num_classes: Number of output classes (offloading decisions)
        """
        self.num_samples = num_samples
        self.num_features = num_features
        self.num_classes = num_classes
        
        # Generate synthetic features and labels
        self.features, self.labels = self._generate_synthetic_data()
        
        logger.info(f"Generated synthetic dataset: {num_samples} samples, "
                   f"{num_features} features, {num_classes} classes")
    
    def _generate_synthetic_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic data with realistic patterns."""
        
        # Feature categories (matching MEC environment state representation):
        # 0-2: User position and mobility
        # 3-4: Current task characteristics (CPU cycles, data size)
        # 5-6: Device resources (battery, computation capacity)
        # 7-9: Network conditions (bandwidth, latency, loss)
        # 10-11: Server loads and distances
        
        features = torch.zeros(self.num_samples, self.num_features)
        
        # User position (normalized coordinates)
        features[:, 0] = torch.rand(self.num_samples)  # x position
        features[:, 1] = torch.rand(self.num_samples)  # y position
        features[:, 2] = torch.randn(self.num_samples) * 0.1  # velocity
        
        # Task characteristics
        features[:, 3] = torch.exponential(torch.ones(self.num_samples))  # CPU cycles (normalized)
        features[:, 4] = torch.exponential(torch.ones(self.num_samples) * 0.5)  # Data size
        
        # Device resources
        features[:, 5] = torch.rand(self.num_samples)  # Battery level
        features[:, 6] = 0.3 + 0.7 * torch.rand(self.num_samples)  # Device capacity
        
        # Network conditions
        features[:, 7] = torch.rand(self.num_samples)  # Bandwidth
        features[:, 8] = torch.exponential(torch.ones(self.num_samples) * 0.5)  # Latency
        features[:, 9] = torch.rand(self.num_samples) * 0.1  # Packet loss
        
        # Server conditions
        features[:, 10] = torch.rand(self.num_samples)  # Server load
        features[:, 11] = torch.rand(self.num_samples)  # Distance to nearest server
        
        # Generate labels based on realistic decision patterns
        labels = self._generate_realistic_labels(features)
        
        return features, labels
    
    def _generate_realistic_labels(self, features: torch.Tensor) -> torch.Tensor:
        """Generate realistic labels based on feature patterns."""
        
        labels = torch.zeros(self.num_samples, dtype=torch.long)
        
        for i in range(self.num_samples):
            # Extract relevant features
            cpu_cycles = features[i, 3].item()
            data_size = features[i, 4].item()
            battery = features[i, 5].item()
            device_capacity = features[i, 6].item()
            bandwidth = features[i, 7].item()
            latency = features[i, 8].item()
            server_load = features[i, 10].item()
            
            # Decision logic (0: local, 1-5: different edge servers)
            
            # Local processing if: low complexity, good battery, good device capacity
            if cpu_cycles < 0.5 and battery > 0.7 and device_capacity > 0.6:
                labels[i] = 0  # Local processing
            
            # Edge processing decisions based on network and server conditions
            elif bandwidth > 0.7 and latency < 0.3 and server_load < 0.5:
                labels[i] = 1  # Primary edge server
            elif bandwidth > 0.5 and latency < 0.5 and server_load < 0.7:
                labels[i] = 2  # Secondary edge server
            elif cpu_cycles > 1.5 or data_size > 1.5:
                labels[i] = 3  # High-capacity server
            elif battery < 0.3:
                labels[i] = 4  # Battery-saving offloading
            else:
                labels[i] = 5  # Default balanced offloading
        
        return labels
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]
    
    def get_feature_statistics(self) -> Dict:
        """Get statistics about generated features."""
        return {
            'feature_means': self.features.mean(dim=0).tolist(),
            'feature_stds': self.features.std(dim=0).tolist(),
            'label_distribution': torch.bincount(self.labels).tolist(),
            'num_samples': self.num_samples,
            'num_features': self.num_features,
            'num_classes': self.num_classes
        }


class FederatedDatasetManager:
    """
    Manages federated dataset distribution across clients.
    Handles non-IID data distribution and client dataset creation.
    """
    
    def __init__(self, total_samples: int = 10000, num_clients: int = 20, 
                 alpha: float = 0.5, batch_size: int = 32):
        """
        Initialize federated dataset manager.
        
        Args:
            total_samples: Total number of samples to generate
            num_clients: Number of federated clients
            alpha: Dirichlet distribution parameter (lower = more non-IID)
            batch_size: Batch size for client data loaders
        """
        self.total_samples = total_samples
        self.num_clients = num_clients
        self.alpha = alpha
        self.batch_size = batch_size
        
        # Generate master dataset
        self.master_dataset = MobileTaskDataset(total_samples)
        
        # Distribute data across clients
        self.client_datasets = self._create_federated_split()
        
        logger.info(f"Created federated datasets for {num_clients} clients "
                   f"with alpha={alpha} (non-IID factor)")
    
    def _create_federated_split(self) -> Dict[int, DataLoader]:
        """Create non-IID federated data split across clients."""
        
        # Get all features and labels
        all_features = self.master_dataset.features
        all_labels = self.master_dataset.labels
        
        # Create non-IID distribution using Dirichlet distribution
        num_classes = self.master_dataset.num_classes
        
        # Group samples by class
        class_indices = {}
        for class_id in range(num_classes):
            class_indices[class_id] = torch.where(all_labels == class_id)[0]
        
        client_datasets = {}
        
        for client_id in range(self.num_clients):
            client_indices = []
            
            # Sample number of samples for this client
            if client_id == self.num_clients - 1:
                # Last client gets remaining samples
                remaining_samples = self.total_samples - sum(len(cd) for cd in client_datasets.values() if hasattr(cd, '__len__'))
                samples_per_client = max(50, remaining_samples)
            else:
                samples_per_client = max(50, int(self.total_samples / self.num_clients * (0.8 + 0.4 * np.random.random())))
            
            # Generate class distribution for this client using Dirichlet
            class_probs = np.random.dirichlet([self.alpha] * num_classes)
            
            for class_id in range(num_classes):
                class_samples = int(samples_per_client * class_probs[class_id])
                available_indices = class_indices[class_id]
                
                if len(available_indices) > 0 and class_samples > 0:
                    # Sample without replacement
                    selected_count = min(class_samples, len(available_indices))
                    selected_indices = available_indices[torch.randperm(len(available_indices))[:selected_count]]
                    client_indices.extend(selected_indices.tolist())
                    
                    # Remove selected indices from available pool
                    mask = torch.ones(len(available_indices), dtype=torch.bool)
                    for idx in selected_indices:
                        mask[available_indices == idx] = False
                    class_indices[class_id] = available_indices[mask]
            
            # Create client dataset
            if client_indices:
                client_features = all_features[client_indices]
                client_labels = all_labels[client_indices]
                client_dataset = TensorDataset(client_features, client_labels)
                client_datasets[client_id] = DataLoader(
                    client_dataset, 
                    batch_size=self.batch_size, 
                    shuffle=True,
                    drop_last=False
                )
            else:
                # Fallback: give client some random samples
                fallback_indices = torch.randperm(len(all_features))[:50]
                client_features = all_features[fallback_indices]
                client_labels = all_labels[fallback_indices]
                client_dataset = TensorDataset(client_features, client_labels)
                client_datasets[client_id] = DataLoader(
                    client_dataset, 
                    batch_size=self.batch_size, 
                    shuffle=True,
                    drop_last=False
                )
        
        return client_datasets
    
    def get_client_dataset(self, client_id: int) -> Optional[DataLoader]:
        """Get dataset for specific client."""
        return self.client_datasets.get(client_id)
    
    def get_validation_dataset(self, validation_split: float = 0.2) -> DataLoader:
        """Create validation dataset from master dataset."""
        total_samples = len(self.master_dataset)
        val_samples = int(total_samples * validation_split)
        
        # Use last samples for validation
        val_indices = list(range(total_samples - val_samples, total_samples))
        val_features = self.master_dataset.features[val_indices]
        val_labels = self.master_dataset.labels[val_indices]
        
        val_dataset = TensorDataset(val_features, val_labels)
        return DataLoader(val_dataset, batch_size=self.batch_size * 2, shuffle=False)
    
    def get_dataset_statistics(self) -> Dict:
        """Get statistics about federated data distribution."""
        stats = {
            'total_samples': self.total_samples,
            'num_clients': self.num_clients,
            'alpha': self.alpha,
            'master_dataset_stats': self.master_dataset.get_feature_statistics(),
            'client_dataset_sizes': {}
        }
        
        for client_id, dataloader in self.client_datasets.items():
            dataset_size = len(dataloader.dataset)
            stats['client_dataset_sizes'][client_id] = dataset_size
        
        stats['min_client_size'] = min(stats['client_dataset_sizes'].values())
        stats['max_client_size'] = max(stats['client_dataset_sizes'].values())
        stats['avg_client_size'] = np.mean(list(stats['client_dataset_sizes'].values()))
        
        return stats


class MobileOffloadingModel(torch.nn.Module):
    """
    Neural network model for mobile task offloading decisions.
    Compatible with the federated learning framework.
    """
    
    def __init__(self, input_dim: int = 12, num_classes: int = 6, hidden_dim: int = 128):
        """
        Initialize mobile offloading model.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of offloading decision classes
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Setup device and move to GPU if available
        self.device = setup_device(preferred_device='cuda')
        
        # Network architecture
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.Dropout(0.3),
            
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_dim // 2),
            torch.nn.Dropout(0.2),
        )
        
        self.classifier = torch.nn.Linear(hidden_dim // 2, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
        # Optimize for GPU
        self = optimize_model_for_gpu(self, self.device)
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


def create_synthetic_federated_datasets(config: Dict) -> Tuple[Dict[int, DataLoader], DataLoader, Dict]:
    """
    Create synthetic federated datasets for training and validation.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (client_datasets, validation_dataset, dataset_stats)
    """
    
    # Extract parameters from config
    num_clients = config['network']['num_mobile_users']
    total_samples = config.get('dataset', {}).get('total_samples', 10000)
    alpha = config.get('dataset', {}).get('non_iid_alpha', 0.5)
    batch_size = config['learning'].get('batch_size', 32)
    
    # Create federated dataset manager
    dataset_manager = FederatedDatasetManager(
        total_samples=total_samples,
        num_clients=num_clients,
        alpha=alpha,
        batch_size=batch_size
    )
    
    # Get client datasets
    client_datasets = {}
    for client_id in range(num_clients):
        client_datasets[client_id] = dataset_manager.get_client_dataset(client_id)
    
    # Get validation dataset
    validation_dataset = dataset_manager.get_validation_dataset()
    
    # Get statistics
    dataset_stats = dataset_manager.get_dataset_statistics()
    
    logger.info(f"Created federated datasets: {num_clients} clients, "
               f"{total_samples} total samples, alpha={alpha}")
    
    return client_datasets, validation_dataset, dataset_stats
