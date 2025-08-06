# Baseline Federated Learning Edge Computing System

## Overview
This is a baseline implementation of a federated learning system for edge computing offloading decisions using NS-3 simulation and PyTorch DQN.

## Architecture
```
Edge Devices (NS-3) --> FL Server (Python) --> DQN Agent
```

## Components

### 1. NS-3 Client (`federated_client.cc`)
- Simulates edge devices in NS-3 environment
- Sends device state (CPU/network utilization) to FL server
- Receives offloading decisions (local/edge/cloud)
- Configurable number of nodes and simulation parameters

### 2. FL Server (`fl_server.py`)
- Simple baseline federated learning server
- Handles multiple client connections
- Uses threshold-based offloading strategy
- Network communication via TCP sockets

### 3. DQN Agent (`simple_dqn.py`)
- Basic Deep Q-Network implementation
- Experience replay buffer
- Epsilon-greedy exploration
- Target network for stable training

### 4. Test Client (`test_client.py`)
- Python test client for server validation
- Sends predefined test states
- Displays server responses

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start FL Server
```bash
python3 fl_server.py
```

### 3. Test with Python Client
```bash
python3 test_client.py
```

### 4. Run NS-3 Simulation
```bash
# Copy federated_client.cc to NS-3 scratch directory
cp federated_client.cc /path/to/ns-3/scratch/
cd /path/to/ns-3/
./ns3 build federated_client
./ns3 run federated_client
```

## Configuration

### Server Configuration
- Host: `0.0.0.0` (all interfaces)
- Port: `12345`
- Timeout: `1.0` seconds

### DQN Parameters
- State dimension: `1` (device utilization)
- Action dimension: `3` (local/edge/cloud)
- Learning rate: `0.001`
- Discount factor: `0.95`

### NS-3 Parameters
- Default nodes: `3`
- Simulation time: `35` seconds
- FL request interval: `2` seconds

## Offloading Strategy

The baseline uses a simple threshold-based strategy:
- State < 30%: Local processing
- State 30-70%: Edge processing  
- State > 70%: Cloud processing

## Next Steps

This baseline provides:
✅ Working client-server communication
✅ Basic FL server architecture
✅ Simple DQN implementation
✅ NS-3 integration template

For enhanced implementation, consider adding:
- CSV data logging
- Advanced FL algorithms
- Performance metrics
- Multi-agent scenarios
- Real network topologies

## Files Structure
```
baseline/
├── federated_client.cc    # NS-3 edge device simulation
├── fl_server.py          # Baseline FL server
├── simple_dqn.py         # Basic DQN agent
├── test_client.py        # Test client
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Enhanced Features Added

### Replay Buffer (`replay_buffer.py`)
- **Simple Replay Buffer**: Basic experience replay for DQN training
- **Prioritized Replay Buffer**: Advanced prioritized experience replay
- **Configurable Capacity**: Adjustable buffer size and sampling
- **Importance Sampling**: Weighted sampling for better learning

### Enhanced DQN (`simple_dqn.py`)
- **Integrated Replay Buffer**: Both simple and prioritized replay
- **Target Network**: Stable Q-learning with periodic updates
- **Reward Function**: Smart reward calculation for offloading decisions
- **Model Persistence**: Save/load trained models
- **Training Statistics**: Comprehensive monitoring and logging

### FL Server with DQN (`fl_server_with_dqn.py`)
- **Intelligent Decisions**: DQN-based offloading decisions
- **Continuous Learning**: Online learning from client interactions
- **Multi-threaded**: Concurrent client handling
- **Model Persistence**: Automatic model saving/loading
- **Training Monitoring**: Real-time training statistics

## Testing Enhanced Features

### Test Replay Buffer
```bash
python3 replay_buffer.py
```

### Test Enhanced DQN
```bash
python3 simple_dqn.py
```

### Run FL Server with DQN
```bash
python3 fl_server_with_dqn.py
```

## Enhanced Architecture
```
Edge Devices (NS-3) --> FL Server with DQN --> Replay Buffer
                           ↓
                     Smart Offloading Decisions
                     (Local/Edge/Cloud)
```

The enhanced baseline now includes:
✅ Experience replay for stable learning
✅ Prioritized replay for efficient training
✅ Smart reward functions for offloading decisions
✅ Continuous online learning from client interactions
✅ Model persistence and training monitoring
