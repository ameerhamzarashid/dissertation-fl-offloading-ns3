"""
Main Script - Federated Learning System Testing and Validation
Tests the implementation with Dueling DQN + PER + MEC Environment
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

# Import GPU utilities first
from utils.gpu_utils import print_gpu_summary, setup_device, clear_gpu_cache

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
from utils.synthetic_data import create_synthetic_federated_datasets, MobileOffloadingModel


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
    print("testing dueling dqn implementation...")
    
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
    
    print("dueling dqn test completed successfully!\n")
    return True


def test_prioritized_replay():
    """Test Prioritized Experience Replay implementation."""
    print("testing prioritized experience replay...")
    
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
    
    print("prioritized experience replay test completed successfully!\n")
    return True


def test_mec_environment():
    """Test MEC Environment implementation."""
    print("testing mec environment...")
    
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
    
    print("mec environment test completed successfully!\n")
    return True


def test_sfea_algorithm():
    """Test SFEA algorithm with synthetic data."""
    print("testing sfea algorithm with synthetic data...")
    try:
        # Load configuration
        config = load_configuration()
        
        # Add dataset configuration
        config['dataset'] = {
            'total_samples': 5000,
            'non_iid_alpha': 0.5,
            'validation_split': 0.2
        }
        config['learning']['batch_size'] = 32
        
        # Create synthetic datasets
        client_datasets, validation_dataset, dataset_stats = create_synthetic_federated_datasets(config)
        
        print(f"Created datasets for {len(client_datasets)} clients")
        print(f"Dataset statistics: {dataset_stats['avg_client_size']:.0f} avg samples per client")
        
        # Create model for offloading decisions
        model = MobileOffloadingModel(
            input_dim=12,  # MEC environment state dimensions
            num_classes=6,  # Offloading decisions (local + 5 edge servers)
            hidden_dim=128
        )
        
        # Initialize SFEA algorithm
        sfea = SFEAFederatedLearning(config, model)
        
        # Run a few rounds of federated training
        config['learning']['global_rounds'] = 3  # Reduced for testing
        config['learning']['local_epochs'] = 2
        
        print("Starting SFEA federated training...")
        results = sfea.run_federated_training(client_datasets, validation_dataset)
        
        print(f"Training completed: {results['total_rounds']} rounds")
        print(f"Communication efficiency: {results['communication_efficiency']}")
        print(f"Final compression ratio: {results['average_compression_ratio']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"sfea algorithm test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ns3_interface():
    """Test NS-3 Interface (Mock) implementation."""
    print("testing ns-3 interface (mock)...")
    
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
    
    print("ns-3 interface test completed successfully!\n")
    return True


def test_integrated_system():
    """Test integrated system with all components including SFEA."""
    print("testing integrated system with sfea...")
    
    # Load configuration
    config = load_configuration()
    
    # Add dataset configuration for testing
    config['dataset'] = {
        'total_samples': 2000,  # Smaller for testing
        'non_iid_alpha': 0.5,
        'validation_split': 0.2
    }
    config['learning']['batch_size'] = 16
    config['network']['num_mobile_users'] = 5  # Reduced for testing
    config['learning']['global_rounds'] = 2
    config['learning']['local_epochs'] = 1
    
    # Setup logging
    experiment_name = f"integrated_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    main_logger, perf_logger, metrics_logger = configure_experiment_logging(
        experiment_name, config, 'logs'
    )
    
    try:
        # Create synthetic datasets
        client_datasets, validation_dataset, dataset_stats = create_synthetic_federated_datasets(config)
        main_logger.info(f"Created synthetic datasets: {dataset_stats['num_clients']} clients")
        
        # Create model
        model = MobileOffloadingModel(input_dim=12, num_classes=6, hidden_dim=64)
        
        # Initialize SFEA system
        sfea = SFEAFederatedLearning(config, model)
        
        # Initialize other components
        mec_env = MECEnvironment(config)
        ns3_interface = MockNS3Interface(config)
        
        # Start NS-3 simulation
        scenario_config = {'num_users': 5, 'num_servers': 5}
        ns3_interface.start_simulation(scenario_config)
        
        main_logger.info("Starting integrated SFEA system test")
        
        # Run federated learning
        start_time = time.time()
        results = sfea.run_federated_training(client_datasets, validation_dataset)
        training_time = time.time() - start_time
        
        # Log results
        main_logger.info(f"SFEA training completed in {training_time:.2f} seconds")
        main_logger.info(f"Results: {results['communication_efficiency']}")
        
        # Get final metrics
        mec_metrics = mec_env.get_performance_metrics()
        ns3_status = ns3_interface.get_simulation_status()
        compression_stats = sfea.get_compression_stats()
        
        print(f"Training rounds: {results['total_rounds']}")
        print(f"Communication savings: {results['communication_efficiency']['compression_savings']:.1%}")
        print(f"Average compression ratio: {results['average_compression_ratio']:.3f}")
        print(f"Total communication cost: {results['total_communication_cost']}")
        
        # Cleanup
        ns3_interface.stop_simulation()
        
        main_logger.info("Integrated SFEA system test completed successfully")
        return True
        
    except Exception as e:
        main_logger.error(f"Integrated system test failed: {e}")
        print(f"Integrated system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run all system tests."""
    
    # Print GPU summary first
    print_gpu_summary()
    
    print("="*60)
    print("federated learning system testing")
    print("federated learning with dueling dqn + per + mec environment")
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
        ("SFEA Algorithm with Synthetic Data", test_sfea_algorithm),
        ("NS-3 Interface", test_ns3_interface),
        ("Integrated System", test_integrated_system)
    ]
    
    for test_name, test_function in tests:
        try:
            print(f"running {test_name} test...")
            result = test_function()
            test_results.append((test_name, result))
            print(f"{test_name} test: {'passed' if result else 'failed'}")
        except Exception as e:
            print(f"{test_name} test failed with error: {e}")
            test_results.append((test_name, False))
        print("-" * 40)
    
    # Summary
    print("\nsystem test results summary:")
    print("="*40)
    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "passed" if result else "failed"
        print(f"{test_name:.<30} {status}")
    
    print("-" * 40)
    print(f"total: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\nall tests passed! implementation is ready.")
        print("system validation completed successfully!")
    else:
        print(f"\n{total_tests - passed_tests} tests failed. please fix issues before proceeding.")
    
    # Clear GPU cache at the end
    clear_gpu_cache()
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
