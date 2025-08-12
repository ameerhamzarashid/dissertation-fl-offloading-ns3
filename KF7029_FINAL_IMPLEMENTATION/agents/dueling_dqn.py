"""
Dueling DQN Implementation
Deep Q-Network with separate value and advantage streams
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from typing import Tuple, Optional
import sys
import os

# Add utils to path for GPU utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.gpu_utils import setup_device, optimize_model_for_gpu, monitor_gpu_usage


class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network architecture that separates state value and action advantage estimation.
    This architecture allows for better learning in scenarios where some actions have similar values.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Initialize the Dueling DQN network.
        
        Args:
            state_dim: Dimensionality of the state space
            action_dim: Number of possible actions
            hidden_dim: Number of neurons in hidden layers
        """
        super(DuelingDQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Shared feature extraction layers
        self.feature_layer1 = nn.Linear(state_dim, hidden_dim)
        self.feature_layer2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Value stream - estimates V(s)
        self.value_layer1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.value_layer2 = nn.Linear(hidden_dim // 2, 1)
        
        # Advantage stream - estimates A(s, a)
        self.advantage_layer1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.advantage_layer2 = nn.Linear(hidden_dim // 2, action_dim)
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
        
        # Set device with GPU optimization
        self.device = setup_device(preferred_device='cuda')
        
        # Optimize model for GPU if available
        self = optimize_model_for_gpu(self, self.device)
        
        # Log GPU info
        gpu_info = monitor_gpu_usage()
        if gpu_info.get('gpu_available'):
            print(f"dueling dqn using gpu: {self.device}")
            print(f"gpu memory allocated: {gpu_info.get('allocated_mb', 0):.1f} mb")
        else:
            print(f"dueling dqn using cpu: {self.device}")
        
        # Training parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.target_update_freq = 100
        self.training_step = 0
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
    def _initialize_weights(self):
        """
        Initialize network weights using Xavier uniform initialization.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the dueling network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Q-values for all actions
        """
        # Shared feature extraction
        features = F.relu(self.feature_layer1(state))
        features = F.relu(self.feature_layer2(features))
        
        # Value stream
        value = F.relu(self.value_layer1(features))
        value = self.value_layer2(value)
        
        # Advantage stream
        advantage = F.relu(self.advantage_layer1(features))
        advantage = self.advantage_layer2(advantage)
        
        # Combine value and advantage using the dueling architecture formula
        # Q(s, a) = V(s) + A(s, a) - mean(A(s, a'))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values
    
    def act(self, state: torch.Tensor, epsilon: Optional[float] = None) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            epsilon: Exploration probability (uses self.epsilon if None)
            
        Returns:
            Selected action index
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        if random.random() > epsilon:
            # Greedy action selection
            with torch.no_grad():
                state = state.to(self.device)
                q_values = self.forward(state)
                action = q_values.argmax().item()
        else:
            # Random action selection
            action = random.randrange(self.action_dim)
        
        return action
    
    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get Q-values for a given state without exploration.
        
        Args:
            state: Input state tensor
            
        Returns:
            Q-values for all actions
        """
        with torch.no_grad():
            state = state.to(self.device)
            return self.forward(state)
    
    def update_epsilon(self):
        """
        Decay epsilon for exploration-exploitation balance.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def compute_loss(self, 
                    states: torch.Tensor, 
                    actions: torch.Tensor, 
                    rewards: torch.Tensor, 
                    next_states: torch.Tensor, 
                    dones: torch.Tensor, 
                    target_network: 'DuelingDQN',
                    gamma: float = 0.99) -> torch.Tensor:
        """
        Compute the DQN loss using target network and double DQN.
        
        Args:
            states: Batch of states
            actions: Batch of actions taken
            rewards: Batch of rewards received
            next_states: Batch of next states
            dones: Batch of done flags
            target_network: Target network for stable learning
            gamma: Discount factor
            
        Returns:
            Computed loss tensor
        """
        # Move tensors to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q-values
        current_q_values = self.forward(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: use current network to select actions, target network to evaluate
        with torch.no_grad():
            # Select actions using current network
            next_q_values_current = self.forward(next_states)
            next_actions = next_q_values_current.argmax(dim=1)
            
            # Evaluate selected actions using target network
            next_q_values_target = target_network.forward(next_states)
            next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # Calculate target Q-values
            target_q_values = rewards + (gamma * next_q_values * (1 - dones))
        
        # Compute loss (Mean Squared Error)
        loss = F.mse_loss(current_q_values, target_q_values)
        
        return loss
    
    def train_step(self, 
                   states: torch.Tensor, 
                   actions: torch.Tensor, 
                   rewards: torch.Tensor, 
                   next_states: torch.Tensor, 
                   dones: torch.Tensor, 
                   target_network: 'DuelingDQN',
                   gamma: float = 0.99) -> float:
        """
        Perform one training step.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            target_network: Target network
            gamma: Discount factor
            
        Returns:
            Loss value
        """
        self.optimizer.zero_grad()
        
        loss = self.compute_loss(states, actions, rewards, next_states, dones, target_network, gamma)
        
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        self.training_step += 1
        
        # Update epsilon
        self.update_epsilon()
        
        return loss.item()
    
    def compute_td_errors(self, 
                         states: torch.Tensor, 
                         actions: torch.Tensor, 
                         rewards: torch.Tensor, 
                         next_states: torch.Tensor, 
                         dones: torch.Tensor, 
                         target_network: 'DuelingDQN',
                         gamma: float = 0.99) -> torch.Tensor:
        """
        Compute TD errors for prioritized experience replay.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            target_network: Target network
            gamma: Discount factor
            
        Returns:
            TD errors for each sample
        """
        with torch.no_grad():
            # Move tensors to device
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)
            
            # Current Q-values
            current_q_values = self.forward(states)
            current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Double DQN target calculation
            next_q_values_current = self.forward(next_states)
            next_actions = next_q_values_current.argmax(dim=1)
            
            next_q_values_target = target_network.forward(next_states)
            next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            target_q_values = rewards + (gamma * next_q_values * (1 - dones))
            
            # TD errors
            td_errors = torch.abs(current_q_values - target_q_values)
            
        return td_errors
    
    def soft_update(self, target_network: 'DuelingDQN', tau: float = 0.005):
        """
        Soft update of target network parameters.
        θ_target = τ * θ_local + (1 - τ) * θ_target
        
        Args:
            target_network: Target network to update
            tau: Interpolation parameter
        """
        for target_param, local_param in zip(target_network.parameters(), self.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def hard_update(self, target_network: 'DuelingDQN'):
        """
        Hard update of target network parameters.
        
        Args:
            target_network: Target network to update
        """
        target_network.load_state_dict(self.state_dict())
    
    def save_model(self, filepath: str):
        """
        Save the model state.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dim': self.hidden_dim
        }, filepath)
    
    def load_model(self, filepath: str):
        """
        Load the model state.
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.training_step = checkpoint.get('training_step', 0)
    
    def get_model_info(self) -> dict:
        """
        Get information about the model architecture and training state.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': 'Dueling DQN',
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dim': self.hidden_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'learning_rate': self.learning_rate
        }
