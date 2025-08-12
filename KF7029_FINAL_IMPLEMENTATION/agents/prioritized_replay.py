"""
Prioritized Experience Replay Buffer
Implements experience replay with prioritization based on TD-error magnitude
"""

import numpy as np
import random
from typing import Tuple, List, Any, Optional
import torch
from collections import namedtuple


# Experience tuple structure
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class SumTree:
    """
    Sum tree data structure for efficient prioritized sampling.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize the sum tree.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float):
        """
        Propagate priority change up the tree.
        
        Args:
            idx: Index in the tree
            change: Change in priority value
        """
        parent = (idx - 1) // 2
        self.tree[parent] += change
        
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """
        Retrieve leaf index based on cumulative sum.
        
        Args:
            idx: Current tree index
            s: Target cumulative sum
            
        Returns:
            Leaf index
        """
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """
        Get total priority sum.
        
        Returns:
            Total sum of all priorities
        """
        return self.tree[0]
    
    def add(self, priority: float, data: Any):
        """
        Add new experience with given priority.
        
        Args:
            priority: Priority value
            data: Experience data
        """
        idx = self.write_idx + self.capacity - 1
        
        self.data[self.write_idx] = data
        self.update(idx, priority)
        
        self.write_idx += 1
        if self.write_idx >= self.capacity:
            self.write_idx = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx: int, priority: float):
        """
        Update priority for a given index.
        
        Args:
            idx: Tree index
            priority: New priority value
        """
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, Any]:
        """
        Get experience based on cumulative sum.
        
        Args:
            s: Target cumulative sum
            
        Returns:
            Tuple of (tree_index, priority, data)
        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer that samples experiences based on their TD-error.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, 
                 beta_increment: float = 0.001, epsilon: float = 1e-6):
        """
        Initialize the prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment: Rate of beta increment towards 1.0
            epsilon: Small constant to prevent zero priorities
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        # Sum tree for efficient sampling
        self.tree = SumTree(capacity)
        
        # Track maximum priority for new experiences
        self.max_priority = 1.0
        
        # Statistics
        self.total_samples = 0
        self.priority_stats = {
            'min_priority': float('inf'),
            'max_priority': 0.0,
            'avg_priority': 0.0
        }
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool, td_error: Optional[float] = None):
        """
        Add experience to the buffer with priority.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
            td_error: TD-error for priority calculation (uses max if None)
        """
        experience = Experience(state, action, reward, next_state, done)
        
        # Calculate priority
        if td_error is not None:
            priority = (abs(td_error) + self.epsilon) ** self.alpha
        else:
            priority = self.max_priority
        
        # Update max priority
        self.max_priority = max(self.max_priority, priority)
        
        # Add to tree
        self.tree.add(priority, experience)
        
        # Update statistics
        self._update_priority_stats(priority)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                              torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        Sample a batch of experiences based on priorities.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, indices)
        """
        if self.tree.n_entries < batch_size:
            raise ValueError(f"Not enough experiences in buffer: {self.tree.n_entries} < {batch_size}")
        
        experiences = []
        priorities = []
        indices = []
        
        # Calculate segment size for stratified sampling
        segment = self.tree.total() / batch_size
        
        for i in range(batch_size):
            # Sample from each segment
            left = segment * i
            right = segment * (i + 1)
            s = random.uniform(left, right)
            
            idx, priority, experience = self.tree.get(s)
            
            experiences.append(experience)
            priorities.append(priority)
            indices.append(idx)
        
        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        max_weight = (self.tree.n_entries * min(sampling_probabilities)) ** (-self.beta)
        weights = (self.tree.n_entries * sampling_probabilities) ** (-self.beta)
        weights = weights / max_weight  # Normalize weights
        
        # Convert to tensors
        states = torch.FloatTensor([exp.state for exp in experiences])
        actions = torch.LongTensor([exp.action for exp in experiences])
        rewards = torch.FloatTensor([exp.reward for exp in experiences])
        next_states = torch.FloatTensor([exp.next_state for exp in experiences])
        dones = torch.BoolTensor([exp.done for exp in experiences])
        weights = torch.FloatTensor(weights)
        
        # Increment beta towards 1.0
        self.beta = min(1.0, self.beta + self.beta_increment)
        self.total_samples += batch_size
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices: List[int], td_errors: torch.Tensor):
        """
        Update priorities for given experiences based on new TD-errors.
        
        Args:
            indices: List of tree indices
            td_errors: New TD-errors for priority calculation
        """
        td_errors_np = td_errors.detach().cpu().numpy()
        
        for idx, td_error in zip(indices, td_errors_np):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            
            # Update max priority
            self.max_priority = max(self.max_priority, priority)
            
            # Update statistics
            self._update_priority_stats(priority)
    
    def _update_priority_stats(self, priority: float):
        """
        Update priority statistics.
        
        Args:
            priority: New priority value
        """
        self.priority_stats['min_priority'] = min(self.priority_stats['min_priority'], priority)
        self.priority_stats['max_priority'] = max(self.priority_stats['max_priority'], priority)
        
        # Running average of priorities
        if self.total_samples > 0:
            alpha_stats = 0.01  # Smoothing factor for running average
            self.priority_stats['avg_priority'] = (
                (1 - alpha_stats) * self.priority_stats['avg_priority'] + 
                alpha_stats * priority
            )
        else:
            self.priority_stats['avg_priority'] = priority
    
    def __len__(self) -> int:
        """
        Get current buffer size.
        
        Returns:
            Number of experiences in buffer
        """
        return self.tree.n_entries
    
    def is_ready(self, min_size: int) -> bool:
        """
        Check if buffer has enough experiences for sampling.
        
        Args:
            min_size: Minimum required buffer size
            
        Returns:
            True if buffer is ready for sampling
        """
        return len(self) >= min_size
    
    def clear(self):
        """
        Clear all experiences from the buffer.
        """
        self.tree = SumTree(self.capacity)
        self.max_priority = 1.0
        self.total_samples = 0
        self.priority_stats = {
            'min_priority': float('inf'),
            'max_priority': 0.0,
            'avg_priority': 0.0
        }
    
    def get_stats(self) -> dict:
        """
        Get buffer statistics.
        
        Returns:
            Dictionary containing buffer statistics
        """
        return {
            'capacity': self.capacity,
            'current_size': len(self),
            'total_samples_drawn': self.total_samples,
            'alpha': self.alpha,
            'beta': self.beta,
            'max_priority': self.max_priority,
            'priority_statistics': self.priority_stats.copy(),
            'utilization': len(self) / self.capacity if self.capacity > 0 else 0.0
        }
    
    def sample_uniform(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                                       torch.Tensor, torch.Tensor]:
        """
        Sample experiences uniformly (for comparison purposes).
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        if self.tree.n_entries < batch_size:
            raise ValueError(f"Not enough experiences in buffer: {self.tree.n_entries} < {batch_size}")
        
        # Random sampling
        indices = random.sample(range(self.tree.n_entries), batch_size)
        experiences = [self.tree.data[i] for i in indices]
        
        # Convert to tensors
        states = torch.FloatTensor([exp.state for exp in experiences])
        actions = torch.LongTensor([exp.action for exp in experiences])
        rewards = torch.FloatTensor([exp.reward for exp in experiences])
        next_states = torch.FloatTensor([exp.next_state for exp in experiences])
        dones = torch.BoolTensor([exp.done for exp in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def get_experience_priorities(self, num_experiences: int = 100) -> List[float]:
        """
        Get priorities of recent experiences for analysis.
        
        Args:
            num_experiences: Number of recent experiences to analyze
            
        Returns:
            List of priority values
        """
        if self.tree.n_entries == 0:
            return []
        
        num_to_check = min(num_experiences, self.tree.n_entries)
        priorities = []
        
        # Get priorities from tree
        start_idx = self.capacity - 1
        for i in range(num_to_check):
            tree_idx = start_idx + i
            if tree_idx < len(self.tree.tree):
                priorities.append(self.tree.tree[tree_idx])
        
        return priorities
