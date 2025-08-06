#!/usr/bin/env python3
"""
Baseline Replay Buffer for Federated Learning DQN
Simple experience replay implementation for edge computing offloading
"""
import random
import numpy as np
from collections import deque
from typing import List, Tuple, Optional

class ReplayBuffer:
    """Simple experience replay buffer for DQN training"""
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
        
    def push(self, state: float, action: int, reward: float, 
             next_state: float, done: bool):
        """
        Add experience to replay buffer
        
        Args:
            state: Current state (device utilization)
            action: Action taken (0=local, 1=edge, 2=cloud)
            reward: Reward received
            next_state: Next state after action
            done: Whether episode is finished
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> Optional[List[Tuple]]:
        """
        Sample random batch of experiences
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Batch of experiences or None if buffer too small
        """
        if len(self.buffer) < batch_size:
            return None
            
        batch = random.sample(self.buffer, batch_size)
        return batch
    
    def __len__(self) -> int:
        """Return current buffer size"""
        return len(self.buffer)
    
    def is_ready(self, min_size: int = 1000) -> bool:
        """Check if buffer has enough experiences for training"""
        return len(self.buffer) >= min_size

class PrioritizedReplayBuffer:
    """Advanced prioritized experience replay buffer"""
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        """
        Initialize prioritized replay buffer
        
        Args:
            capacity: Maximum buffer size
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.position = 0
        
    def push(self, state: float, action: int, reward: float,
             next_state: float, done: bool, td_error: float = 1.0):
        """
        Add experience with priority
        
        Args:
            state, action, reward, next_state, done: Experience tuple
            td_error: TD error for prioritization
        """
        priority = (abs(td_error) + 1e-6) ** self.alpha
        
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority
            self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4):
        """
        Sample batch with prioritized sampling
        
        Args:
            batch_size: Number of experiences to sample
            beta: Importance sampling exponent
            
        Returns:
            Batch, importance weights, and indices
        """
        if len(self.buffer) < batch_size:
            return None, None, None
            
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Get experiences
        batch = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights = weights / weights.max()  # Normalize
        
        return batch, weights, indices
    
    def update_priorities(self, indices: List[int], td_errors: List[float]):
        """Update priorities based on new TD errors"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
    
    def __len__(self) -> int:
        return len(self.buffer)

def test_replay_buffer():
    """Test function for replay buffer"""
    print("Testing Replay Buffer...")
    
    # Test simple replay buffer
    buffer = ReplayBuffer(capacity=1000)
    
    # Add some experiences
    for i in range(100):
        state = np.random.uniform(0, 100)  # Device utilization
        action = np.random.randint(0, 3)   # Offloading decision
        reward = np.random.uniform(-1, 1)  # Reward
        next_state = np.random.uniform(0, 100)
        done = i % 10 == 0  # Episode ends every 10 steps
        
        buffer.push(state, action, reward, next_state, done)
    
    print(f"Buffer size: {len(buffer)}")
    print(f"Buffer ready for training: {buffer.is_ready(50)}")
    
    # Sample a batch
    batch = buffer.sample(32)
    if batch:
        print(f"Sampled batch size: {len(batch)}")
        print(f"Sample experience: {batch[0]}")
    
    # Test prioritized replay buffer
    print("\nTesting Prioritized Replay Buffer...")
    priority_buffer = PrioritizedReplayBuffer(capacity=1000)
    
    # Add experiences with varying TD errors
    for i in range(100):
        state = np.random.uniform(0, 100)
        action = np.random.randint(0, 3)
        reward = np.random.uniform(-1, 1)
        next_state = np.random.uniform(0, 100)
        done = i % 10 == 0
        td_error = np.random.uniform(0, 2)  # Simulated TD error
        
        priority_buffer.push(state, action, reward, next_state, done, td_error)
    
    # Sample prioritized batch
    batch, weights, indices = priority_buffer.sample(32)
    if batch:
        print(f"Prioritized batch size: {len(batch)}")
        print(f"Importance weights shape: {weights.shape}")
        print(f"Sample indices: {indices[:5]}")
    
    print("Replay buffer tests completed!")

if __name__ == '__main__':
    test_replay_buffer()
