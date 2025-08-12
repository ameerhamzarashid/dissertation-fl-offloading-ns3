"""
Main Script - Day 1 Testing and Validation
Tests the baseline implementation with Dueling DQN + PER + MEC Environment
"""

import os
import sys
import yaml
import torch
import numpy as np
import time
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from core.fl_coordinator import FLCoordinator
from core.sfea_algorithm import SFEAFederatedLearning
from core.client_manager import ClientManager
from core.server_aggregator import ServerAggregator
from agents.dueling_dqn import DuelingDQN
from agents.prioritized_replay import PrioritizedReplayBuffer
from simulation.mec_environment import MECEnvironment
from simulation.ns3_interface import MockNS3Interface
from utils.logger import configure_experiment_logging
from utils.file_manager import FileManager
from utils.communication_tracker import CommunicationTracker


def load_configuration():
    """Load configuration from YAML files."""
    config = {}
    
    # Load base configuration
    with open('configs/base_config.yaml', 'r') as f:
        config.update(yaml.safe_load(f))
    
    # Load network parameters
    with open('configs/network_params.yaml', 'r') as f:
        network_config = yaml.safe_load(f)
        config['ns3'] = network_config
    
    return config


def test_dueling_dqn():
    """Test Dueling DQN implementation."""
    print("Testing Dueling DQN Implementation...")
    
    # Create DQN with test parameters
    state_dim = 12
    action_dim = 6
    hidden_dim = 128
    
    # Initialize networks
    main_network = DuelingDQN(state_dim, action_dim, hidden_dim)
    target_network = DuelingDQN(state_dim, action_dim, hidden_dim)
    
    # Test forward pass
    test_state = torch.randn(1, state_dim)
    q_values = main_network(test_state)
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Q-values shape: {q_values.shape}")
    print(f"Q-values: {q_values.detach().numpy()}")
    
    # Test action selection
    action = main_network.act(test_state, epsilon=0.1)
    print(f"Selected action: {action}")
    
    # Test model info
    model_info = main_network.get_model_info()
    print(f"Model info: {model_info}")
    
    print("Dueling DQN test completed successfully!\n")
    return True


def test_prioritized_replay():
    """Test Prioritized Experience Replay implementation."""
    print("Testing Prioritized Experience Replay...")
    
    # Initialize replay buffer
    buffer = PrioritizedReplayBuffer(
        capacity=1000,
        alpha=0.6,
        beta=0.4
    )
    
    # Add some experiences
    state_dim = 12
    for i in range(50):
        state = np.random.randn(state_dim)
        action = np.random.randint(0, 6)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        done = np.random.random() < 0.1
        td_error = np.random.random() * 2.0
        
        buffer.add(state, action, reward, next_state, done, td_error)
    
    print(f"Buffer size: {len(buffer)}")
    print(f"Buffer ready: {buffer.is_ready(32)}")
    
    # Test sampling
    if len(buffer) >= 32:
        states, actions, rewards, next_states, dones, weights, indices = buffer.sample(32)
        print(f"Sampled batch shapes:")
        print(f"  States: {states.shape}")
        print(f"  Actions: {actions.shape}")
        print(f"  Rewards: {rewards.shape}")
        print(f"  Weights: {weights.shape}")
        print(f"  Indices length: {len(indices)}")
    
    # Test statistics
    stats = buffer.get_stats()
    print(f"Buffer statistics: {stats}")
    
    print("Prioritized Experience Replay test completed successfully!\n")
    return True


def test_mec_environment():
    """Test MEC Environment implementation."""
    print("Testing MEC Environment...")
    
    # Load configuration
    config = load_configuration()
    
    # Initialize environment
    env = MECEnvironment(config)
    
    # Test reset
    state = env.reset(user_id=0)
    print(f"Initial state shape: {state.shape}")
    print(f"Initial state: {state}")
    
    # Test steps
    total_reward = 0.0
    for step in range(10):
        action = np.random.randint(0, 6)  # Random action
        next_state, reward, done = env.step(action, user_id=0)
        total_reward += reward
        
        print(f"Step {step}: action={action}, reward={reward:.4f}, done={done}")
        
        if done:
            break
    
    print(f"Total reward: {total_reward:.4f}")
    
    # Test performance metrics
    metrics = env.get_performance_metrics()
    print(f"Performance metrics: {metrics}")
    
    print("MEC Environment test completed successfully!\n")
    return True


def test_fl_coordinator():
    """Test FL Coordinator implementation."""
    print("Testing FL Coordinator...")
    
    # Load configuration
    config = load_configuration()
    
    # Initialize coordinator
    coordinator = FLCoordinator(config)
    
    # Initialize clients
    coordinator.initialize_clients()
    
    # Test model distribution
    client_models = coordinator.distribute_global_model()
    print(f"Distributed models to {len(client_models)} clients")
    
    # Simulate client updates
    client_updates = {}
    for client_id in range(config['network']['num_mobile_users']):
        # Create dummy update (random gradients)
        update = {}
        for name, param in coordinator.global_model.named_parameters():
            update[name] = torch.randn_like(param) * 0.01
        client_updates[client_id] = update
    
    # Test aggregation
    coordinator.collect_client_updates(client_updates)
    updated_model = coordinator.aggregate_models()
    
    print(f"Aggregated updates from {len(client_updates)} clients")
    
    # Test training summary
    summary = coordinator.get_training_summary()
    print(f"Training summary: {summary}")
    
    print("FL Coordinator test completed successfully!\n")
    return True


def test_ns3_interface():
    """Test NS-3 Interface (Mock) implementation."""
    print("Testing NS-3 Interface (Mock)...")
    
    # Load configuration
    config = load_configuration()
    
    # Initialize mock interface
    ns3_interface = MockNS3Interface(config)
    
    # Test simulation start
    scenario_config = {
        'num_users': 10,
        'num_servers': 5,
        'output_directory': 'test_results'
    }
    
    success = ns3_interface.start_simulation(scenario_config)
    print(f"Simulation started: {success}")
    
    if success:
        # Wait for some metrics
        import time
        time.sleep(3)
        
        # Get topology and metrics
        topology = ns3_interface.get_network_topology()
        metrics = ns3_interface.get_network_metrics(last_n=5)
        status = ns3_interface.get_simulation_status()
        
        print(f"Topology: {topology}")
        print(f"Recent metrics count: {len(metrics)}")
        print(f"Simulation status: {status}")
        
        # Stop simulation
        ns3_interface.stop_simulation()
    
    print("NS-3 Interface test completed successfully!\n")
    return True


def test_integrated_system():
    """Test integrated system with all components."""
    print("Testing Integrated System...")
    
    # Load configuration
    config = load_configuration()
    
    # Reduce scale for testing
    config['network']['num_mobile_users'] = 5
    config['learning']['global_rounds'] = 3
    config['learning']['local_epochs'] = 2
    
    # Setup logging
    experiment_name = f"day1_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    main_logger, perf_logger, metrics_logger = configure_experiment_logging(
        experiment_name, config, 'logs'
    )
    
    # Initialize components
    coordinator = FLCoordinator(config)
    mec_env = MECEnvironment(config)
    ns3_interface = MockNS3Interface(config)
    
    # Start NS-3 simulation
    scenario_config = {'num_users': 5, 'num_servers': 5}
    ns3_interface.start_simulation(scenario_config)
    
    # Initialize FL clients
    coordinator.initialize_clients()
    
    main_logger.info("Starting integrated system test")
    
    # Run a few federated learning rounds
    for round_num in range(config['learning']['global_rounds']):
        start_time = time.time()
        
        # Run federated round
        round_metrics = coordinator.run_federated_round(round_num, mec_env)
        
        # Log performance
        round_time = time.time() - start_time
        perf_logger.log_timing(f"FL_Round_{round_num}", round_time)
        metrics_logger.log_training_metrics(round_num, round_metrics)
        
        main_logger.info(f"Round {round_num} completed: {round_metrics}")
    
    # Get final results
    training_summary = coordinator.get_training_summary()
    mec_metrics = mec_env.get_performance_metrics()
    ns3_status = ns3_interface.get_simulation_status()
    
    print(f"Training Summary: {training_summary}")
    print(f"MEC Metrics: {mec_metrics}")
    print(f"NS-3 Status: {ns3_status}")
    
    # Cleanup
    ns3_interface.stop_simulation()
    
    main_logger.info("Integrated system test completed successfully")
    print("Integrated System test completed successfully!\n")
    return True


def main():
    """Main function to run all Day 1 tests."""
    print("="*60)
    print("DAY 1 IMPLEMENTATION TESTING")
    print("Federated Learning with Dueling DQN + PER + MEC Environment")
    print("="*60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    test_results = []
    
    # Run individual component tests
    tests = [
        ("Dueling DQN", test_dueling_dqn),
        ("Prioritized Experience Replay", test_prioritized_replay),
        ("MEC Environment", test_mec_environment),
        ("FL Coordinator", test_fl_coordinator),
        ("NS-3 Interface", test_ns3_interface),
        ("Integrated System", test_integrated_system)
    ]
    
    for test_name, test_function in tests:
        try:
            print(f"Running {test_name} test...")
            result = test_function()
            test_results.append((test_name, result))
            print(f"{test_name} test: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"{test_name} test FAILED with error: {e}")
            test_results.append((test_name, False))
        print("-" * 40)
    
    # Summary
    print("\nDAY 1 TEST RESULTS SUMMARY:")
    print("="*40)
    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:.<30} {status}")
    
    print("-" * 40)
    print(f"Total: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\nALL TESTS PASSED! Day 1 implementation is ready.")
        print("Proceed to Day 2: SFEA Algorithm Development")
    else:
        print(f"\n{total_tests - passed_tests} tests failed. Please fix issues before proceeding.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
