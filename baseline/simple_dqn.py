#!/usr/bin/env python3
"""
Baseline DQN Agent for Federated Learning Offloading
Implementation with replay buffer for edge computing decision making
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

class SimpleDQN(nn.Module):
    """Simple DQN network for offloading decisions"""
    
    def __init__(self, state_dim: int = 1, action_dim: int = 3, hidden_dim: int = 64):
        super(SimpleDQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    """DQN Agent with replay buffer for offloading decisions"""
    
    def __init__(self, 
                 state_dim: int = 1,
                 action_dim: int = 3,
                 lr: float = 0.001,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 memory_size: int = 10000,
                 batch_size: int = 32,
                 use_prioritized: bool = False):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.use_prioritized = use_prioritized
        
        # Neural networks
        self.q_network = SimpleDQN(state_dim, action_dim)
        self.target_network = SimpleDQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay buffer
        if use_prioritized:
            self.memory = PrioritizedReplayBuffer(memory_size)
        else:
            self.memory = ReplayBuffer(memory_size)
        
        # Training statistics
        self.training_step = 0
        self.target_update_freq = 100
        
        # Update target network
        self.update_target_network()
        
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        if self.use_prioritized:
            # For prioritized replay, we'll update TD error later
            self.memory.push(state, action, reward, next_state, done, td_error=1.0)
        else:
            self.memory.push(state, action, reward, next_state, done)
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        state_tensor = torch.FloatTensor([state]).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if not self.memory.is_ready(self.batch_size):
            return
        
        self.training_step += 1
        
        if self.use_prioritized:
            batch, weights, indices = self.memory.sample(self.batch_size)
            if batch is None:
                return
                
            weights = torch.FloatTensor(weights)
        else:
            batch = self.memory.sample(self.batch_size)
            if batch is None:
                return
                
            weights = torch.ones(self.batch_size)
            indices = None
        
        # Convert batch to tensors
        states = torch.FloatTensor([[e[0]] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([[e[3]] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # TD errors for prioritized replay
        td_errors = target_q_values - current_q_values.squeeze()
        
        # Weighted loss
        loss = (weights * td_errors.pow(2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update priorities if using prioritized replay
        if self.use_prioritized and indices is not None:
            td_errors_abs = td_errors.abs().detach().cpu().numpy()
            self.memory.update_priorities(indices, td_errors_abs)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Update target network periodically
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()
    
    def compute_reward(self, state, action, next_state):
        """
        Compute reward for offloading decision
        
        Args:
            state: Current device utilization (0-100)
            action: Offloading decision (0=local, 1=edge, 2=cloud)
            next_state: Next device utilization
            
        Returns:
            reward: Calculated reward value
        """
        # Simple reward function based on utilization and action
        utilization = state / 100.0  # Normalize to 0-1
        
        if action == 0:  # Local processing
            # Reward high for low utilization, penalty for high utilization
            reward = 1.0 - utilization
        elif action == 1:  # Edge processing
            # Balanced reward for medium utilization
            reward = 0.5 if 0.3 <= utilization <= 0.7 else 0.2
        else:  # Cloud processing
            # Reward high for high utilization
            reward = utilization
        
        # Add small penalty for switching actions frequently
        state_change = abs(next_state - state) / 100.0
        reward -= 0.1 * state_change
        
        return reward
    
    def save(self, filepath):
        """Save model and replay buffer"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'memory_size': len(self.memory)
        }, filepath)
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint.get('training_step', 0)

def test_dqn_with_replay():
    """Test DQN agent with replay buffer"""
    print("Testing DQN Agent with Replay Buffer...")
    
    # Test both simple and prioritized replay
    for use_prioritized in [False, True]:
        buffer_type = "Prioritized" if use_prioritized else "Simple"
        print(f"Testing with {buffer_type} Replay Buffer")
        
        agent = DQNAgent(use_prioritized=use_prioritized)
        
        # Simulate training episodes
        for episode in range(20):
            state = np.random.uniform(0, 100)  # Random device utilization
            
            for step in range(10):
                # Agent chooses action
                action = agent.act(state)
                
                # Simulate environment step
                next_state = np.random.uniform(0, 100)
                reward = agent.compute_reward(state, action, next_state)
                done = step == 9  # Episode ends after 10 steps
                
                # Store experience
                agent.remember(state, action, reward, next_state, done)
                
                # Train if enough experiences
                if len(agent.memory) > agent.batch_size:
                    agent.replay()
                
                state = next_state
            
            if episode % 5 == 0:
                print(f"Episode {episode}: Buffer size={len(agent.memory)}, "
                      f"Epsilon={agent.epsilon:.3f}, "
                      f"Training steps={agent.training_step}")
        
        print(f"Final buffer size: {len(agent.memory)}")
        print(f"Final epsilon: {agent.epsilon:.3f}")
    
    print("\nDQN with replay buffer test completed!")

if __name__ == '__main__':
    test_dqn_with_replay()
