#!/usr/bin/env python3
"""
Baseline DQN Agent for Federated Learning Offloading
Simple implementation for edge computing decision making
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

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
    """Simple DQN Agent for offloading decisions"""
    
    def __init__(self, 
                 state_dim: int = 1,
                 action_dim: int = 3,
                 lr: float = 0.001,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 memory_size: int = 10000,
                 batch_size: int = 32):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Neural networks
        self.q_network = SimpleDQN(state_dim, action_dim)
        self.target_network = SimpleDQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Update target network
        self.update_target_network()
        
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']

def test_dqn():
    """Simple test of DQN agent"""
    print("Testing Simple DQN Agent...")
    
    agent = DQNAgent()
    
    # Test with some dummy data
    for episode in range(10):
        state = [np.random.uniform(0, 100)]  # Random device state
        action = agent.act(state)
        reward = np.random.uniform(-1, 1)    # Random reward
        next_state = [np.random.uniform(0, 100)]
        done = episode == 9
        
        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        
        print(f"Episode {episode}: State={state[0]:.2f}, Action={action}, Reward={reward:.2f}, Epsilon={agent.epsilon:.3f}")
    
    print("DQN test completed!")

if __name__ == '__main__':
    test_dqn()
