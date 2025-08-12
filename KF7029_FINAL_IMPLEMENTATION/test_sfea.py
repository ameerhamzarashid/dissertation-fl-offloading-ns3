"""
SFEA Implementation Test
Tests the enhanced federated learning with gradient sparsification
"""

import sys
import os
import yaml
import numpy as np
import torch
import torch.nn as nn

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def test_gradient_sparsification():
    """Test gradient sparsification functionality."""
    print("Testing Gradient Sparsification...")
    try:
        from core.gradient_sparsification import TopKSparsifier, GradientCompressor
        
        # Create test gradients
        test_gradients = {
            'layer1.weight': torch.randn(10, 5),
            'layer1.bias': torch.randn(10),
            'layer2.weight': torch.randn(3, 10),
            'layer2.bias': torch.randn(3)
        }
        
        # Test Top-k sparsifier
        sparsifier = TopKSparsifier(sparsity_ratio=0.2, accumulate_error=True)
        sparse_grads, info = sparsifier.sparsify_gradients(test_gradients, client_id="test_client")
        
        print(f"Original size: {info['original_size']}")
        print(f"Sparse size: {info['sparse_size']}")
        print(f"Compression ratio: {info['compression_ratio']:.3f}")
        
        # Test gradient compressor
        compression_config = {
            'type': 'topk',
            'sparsity_ratio': 0.1,
            'enabled': True
        }
        compressor = GradientCompressor(compression_config)
        compressed_grads, comp_info = compressor.compress(test_gradients)
        
        print(f"Compression successful: {comp_info['compression_ratio']:.3f}")
        return True
        
    except Exception as e:
        print(f"Gradient sparsification test failed: {e}")
        return False

def test_sfea_algorithm_setup():
    """Test SFEA algorithm initialization."""
    print("Testing SFEA Algorithm Setup...")
    try:
        from core.sfea_algorithm import SFEAFederatedLearning
        
        # Load configuration
        with open('configs/sfea_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Create simple test model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 5)
                self.fc2 = nn.Linear(5, 2)
            
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                return self.fc2(x)
        
        model = SimpleModel()
        
        # Initialize SFEA
        sfea = SFEAFederatedLearning(config, model)
        
        print(f"SFEA initialized with {sfea.num_clients} clients")
        print(f"Compression enabled: {sfea.gradient_compressor.compression_enabled}")
        print(f"Global rounds: {sfea.num_rounds}")
        
        # Test compression stats
        stats = sfea.get_compression_stats()
        print(f"Initial compression stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"SFEA algorithm setup test failed: {e}")
        return False

def test_communication_tracker():
    """Test communication efficiency tracking."""
    print("Testing Communication Tracker...")
    try:
        from utils.communication_tracker import CommunicationTracker
        
        # Initialize tracker
        tracker = CommunicationTracker("test_experiment", "logs")
        
        # Simulate round tracking
        tracker.start_round_tracking(round_num=0, num_clients=3)
        
        # Simulate client communications
        for client_id in range(3):
            comm_info = {
                'bytes_sent': 1000 + client_id * 100,
                'bytes_original': 2000,
                'compression_ratio': 0.5 + client_id * 0.1,
                'transmission_time': 0.5,
                'network_condition': {
                    'bandwidth': 5.0,
                    'latency': 50,
                    'loss_rate': 0.01
                }
            }
            tracker.track_client_communication(0, str(client_id), comm_info)
        
        # End round tracking
        round_results = tracker.end_round_tracking(0)
        print(f"Round results: {round_results}")
        
        # Get efficiency metrics
        efficiency = tracker.get_bandwidth_efficiency()
        print(f"Bandwidth efficiency: {efficiency}")
        
        # Generate summary
        summary = tracker.get_summary_report()
        print("Summary report generated successfully")
        
        return True
        
    except Exception as e:
        print(f"Communication tracker test failed: {e}")
        return False

def test_sfea_configuration():
    """Test SFEA configuration loading."""
    print("Testing SFEA Configuration...")
    try:
        # Load SFEA config
        with open('configs/sfea_config.yaml', 'r') as f:
            sfea_config = yaml.safe_load(f)
        
        # Verify key sections
        required_sections = [
            'compression', 'learning', 'communication', 
            'dqn', 'environment', 'network', 'sfea'
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in sfea_config:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"Missing sections: {missing_sections}")
            return False
        
        print(f"Compression enabled: {sfea_config['compression']['enabled']}")
        print(f"Sparsity ratio: {sfea_config['compression']['sparsity_ratio']}")
        print(f"Global rounds: {sfea_config['learning']['global_rounds']}")
        print(f"Communication budget: {sfea_config['communication']['budget_limit']}")
        
        # Test scenarios configuration
        scenarios = sfea_config['experiment']['scenarios']
        print(f"Test scenarios available: {len(scenarios)}")
        for scenario in scenarios:
            print(f"  - {scenario['name']}: {scenario['description']}")
        
        return True
        
    except Exception as e:
        print(f"SFEA configuration test failed: {e}")
        return False

def test_integration_readiness():
    """Test if all components are ready for integration."""
    print("Testing Integration Readiness...")
    try:
        # Test all imports
        from core.gradient_sparsification import TopKSparsifier, AdaptiveSparsifier, GradientCompressor
        from core.sfea_algorithm import SFEAFederatedLearning
        from utils.communication_tracker import CommunicationTracker
        
        # Test configuration compatibility
        with open('configs/base_config.yaml', 'r') as f:
            base_config = yaml.safe_load(f)
        
        with open('configs/sfea_config.yaml', 'r') as f:
            sfea_config = yaml.safe_load(f)
        
        # Merge configurations
        merged_config = {**base_config, **sfea_config}
        
        print("All SFEA components imported successfully")
        print("Configuration files loaded and merged")
        print(f"Merged config sections: {list(merged_config.keys())}")
        
        # Test basic compatibility
        if 'compression' in merged_config and 'learning' in merged_config:
            print("Core SFEA sections present")
        
        if 'dqn' in merged_config and 'environment' in merged_config:
            print("DQN and environment configurations compatible")
        
        return True
        
    except Exception as e:
        print(f"Integration readiness test failed: {e}")
        return False

def main():
    """Run SFEA implementation tests."""
    print("="*60)
    print("SFEA IMPLEMENTATION TEST")
    print("Enhanced Federated Learning with Gradient Sparsification")
    print("="*60)
    
    tests = [
        ("SFEA Configuration", test_sfea_configuration),
        ("Gradient Sparsification", test_gradient_sparsification),
        ("SFEA Algorithm Setup", test_sfea_algorithm_setup),
        ("Communication Tracker", test_communication_tracker),
        ("Integration Readiness", test_integration_readiness)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"{test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"{test_name} failed with exception: {e}")
            results.append((test_name, False))
        print("-" * 40)
    
    # Summary
    print("\nSFEA TEST RESULTS:")
    print("="*40)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:.<35} {status}")
    
    print("-" * 40)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nSFEA IMPLEMENTATION READY!")
        print("Enhanced federated learning components operational")
        print("Gradient sparsification system functional")
        print("Communication optimization ready")
        print("Ready for full system testing and evaluation")
    else:
        print(f"\n{total - passed} components need attention")
        print("Fix issues before proceeding to system evaluation")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    print(f"\nSFEA test completed. Status: {'SUCCESS' if success else 'NEEDS_FIXES'}")
