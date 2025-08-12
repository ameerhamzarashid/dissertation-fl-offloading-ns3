# SFEA Federated Learning Implementation

## Overview
This repository contains the complete implementation of **Sparsification-based Federated Edge-AI (SFEA)** algorithm for efficient federated learning in mobile edge computing environments.

## Key Features

### 🚀 Core Components
- **Federated Learning Framework**: Complete FL coordination with client-server architecture
- **Dueling DQN**: Advanced reinforcement learning with separated value/advantage streams
- **Prioritized Experience Replay**: TD-error based experience sampling for improved learning
- **MEC Environment Simulation**: Realistic edge computing environment with task generation
- **NS-3 Integration**: Network simulation interface for realistic communication modeling

### 📡 SFEA Algorithm Features
- **Gradient Sparsification**: Top-k compression with configurable ratios (10%, 20%, 50%)
- **Error Accumulation**: Gradient preservation mechanism to maintain learning quality
- **Adaptive Compression**: Dynamic compression based on network conditions
- **Communication Optimization**: 50-90% bandwidth reduction achieved
- **Real-time Monitoring**: Communication efficiency tracking and analysis

### ⚙️ Configuration Management
- **Multiple Scenarios**: Baseline, compressed (10%, 20%, 50%), and adaptive compression
- **Flexible Parameters**: Comprehensive configuration system for all components
- **Experimental Setup**: Ready-to-run experimental configurations

## Repository Structure

```
KF7029_FINAL_IMPLEMENTATION/
├── core/                           # Core federated learning components
│   ├── fl_coordinator.py          # Main FL coordination logic
│   ├── sfea_algorithm.py          # SFEA algorithm implementation
│   ├── gradient_sparsification.py # Gradient compression system
│   ├── client_manager.py          # Client management
│   └── server_aggregator.py       # Server-side aggregation
├── agents/                         # Deep reinforcement learning
│   ├── dueling_dqn.py             # Dueling DQN implementation
│   └── prioritized_replay.py      # Prioritized experience replay
├── simulation/                     # Environment simulation
│   ├── mec_environment.py         # MEC environment modeling
│   └── ns3_interface.py           # NS-3 network simulation
├── utils/                          # Utilities and monitoring
│   ├── logger.py                  # Comprehensive logging system
│   ├── file_manager.py            # File management utilities
│   └── communication_tracker.py   # Communication efficiency tracking
├── configs/                        # Configuration files
│   ├── base_config.yaml          # Base configuration
│   ├── network_params.yaml       # Network parameters
│   └── sfea_config.yaml          # SFEA-specific settings
├── main.py                        # Main execution script
├── validate.py                    # Implementation validation
├── test_sfea.py                   # SFEA component testing
└── requirements.txt               # Python dependencies
```

## Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/ameerhamzarashid/dissertation-fl-offloading-ns3.git
cd dissertation-fl-offloading-ns3
git checkout final-implementation

# Navigate to implementation directory
cd KF7029_FINAL_IMPLEMENTATION

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Validation
```bash
# Validate implementation structure
python validate.py

# Test SFEA components
python test_sfea.py
```

### 3. Run Experiments
```bash
# Run complete federated learning with SFEA
python main.py
```

## Configuration

### Basic Configuration (configs/base_config.yaml)
- Learning parameters (rounds, epochs, learning rate)
- Network topology (users, servers, area)
- DQN parameters (architecture, replay buffer)
- Environment settings (task generation, mobility)

### SFEA Configuration (configs/sfea_config.yaml)
- Gradient compression settings
- Communication optimization parameters
- Experimental scenarios
- Performance monitoring settings

## Experimental Scenarios

The implementation supports multiple pre-configured scenarios:

1. **Baseline**: Standard federated learning without compression
2. **SFEA-10%**: Top-k sparsification with 10% gradient retention
3. **SFEA-20%**: Top-k sparsification with 20% gradient retention  
4. **SFEA-50%**: Top-k sparsification with 50% gradient retention
5. **Adaptive**: Dynamic compression based on network conditions

## Performance Metrics

The system tracks and reports:
- **Communication Efficiency**: Bandwidth usage and compression ratios
- **Learning Performance**: Convergence speed and model accuracy
- **System Performance**: Resource usage and computation time
- **Network Impact**: Latency, throughput, and packet loss effects

## Key Algorithms

### SFEA Gradient Sparsification
1. Compute local gradients from client training
2. Apply Top-k selection based on gradient magnitude
3. Accumulate pruned gradients as error compensation
4. Transmit only selected gradients to server
5. Server aggregates sparse gradients with error correction

### Communication Optimization
- Real-time network condition monitoring
- Adaptive compression ratio adjustment
- Communication budget management
- Quality threshold enforcement

## Research Impact

This implementation demonstrates:
- **50-90% bandwidth reduction** through gradient sparsification
- **Maintained learning quality** with error accumulation
- **Scalable federated learning** for edge computing environments
- **Adaptive communication** based on network conditions

## Dependencies

Core requirements:
- Python 3.8+
- PyTorch 1.9+
- NumPy, SciPy
- PyYAML for configuration
- Additional utilities (see requirements.txt)

## Testing

Comprehensive testing framework included:
- Structure validation (`validate.py`)
- Component testing (`test_sfea.py`)
- Integration testing (`main.py`)
- Performance benchmarking

## Authors

Developed as part of dissertation research on federated learning optimization for mobile edge computing environments.

## License

This implementation is part of academic research. Please cite appropriately if used in publications.
