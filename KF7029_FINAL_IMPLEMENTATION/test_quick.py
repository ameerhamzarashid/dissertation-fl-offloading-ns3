"""
Quick component test to verify Day 1 implementation
"""

import sys
import os
import torch
import numpy as np
import yaml

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def test_dueling_dqn():
    """Quick test of Dueling DQN."""
    print("Testing Dueling DQN...")
    try:
        from agents.dueling_dqn import DuelingDQN
        
        # Create DQN
        dqn = DuelingDQN(state_dim=12, action_dim=6, hidden_dim=128)
        
        # Test forward pass
        test_state = torch.randn(1, 12)
        q_values = dqn(test_state)
        
        print(f"✓ DQN output shape: {q_values.shape}")
        print(f"✓ DQN forward pass successful")
        
        # Test action selection
        action = dqn.act(test_state, epsilon=0.1)
        print(f"✓ Action selection: {action}")
        
        return True
    except Exception as e:
        print(f"✗ DQN test failed: {e}")
        return False

def test_replay_buffer():
    """Quick test of Prioritized Replay Buffer."""
    print("Testing Prioritized Replay Buffer...")
    try:
        from agents.prioritized_replay import PrioritizedReplayBuffer
        
        # Create buffer
        buffer = PrioritizedReplayBuffer(capacity=1000, alpha=0.6, beta=0.4)
        
        # Add experience
        state = np.random.randn(12)
        action = 1
        reward = 0.5
        next_state = np.random.randn(12)
        done = False
        td_error = 0.1
        
        buffer.add(state, action, reward, next_state, done, td_error)
        
        print(f"✓ Buffer size: {len(buffer)}")
        print(f"✓ Experience added successfully")
        
        return True
    except Exception as e:
        print(f"✗ Replay buffer test failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading."""
    print("Testing configuration loading...")
    try:
        # Load base config
        with open('configs/base_config.yaml', 'r') as f:
            base_config = yaml.safe_load(f)
        
        # Load network config
        with open('configs/network_params.yaml', 'r') as f:
            network_config = yaml.safe_load(f)
        
        print(f"✓ Base config loaded: {len(base_config)} sections")
        print(f"✓ Network config loaded: {len(network_config)} sections")
        
        return True
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False

def test_mec_environment():
    """Quick test of MEC Environment."""
    print("Testing MEC Environment...")
    try:
        # Load config first
        with open('configs/base_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        from simulation.mec_environment import MECEnvironment
        
        # Create environment
        env = MECEnvironment(config)
        
        # Test reset
        state = env.reset(user_id=0)
        print(f"✓ Environment reset, state shape: {state.shape}")
        
        # Test step
        action = 1  # Local processing
        next_state, reward, done = env.step(action, user_id=0)
        print(f"✓ Environment step successful, reward: {reward:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ MEC Environment test failed: {e}")
        return False

def main():
    """Run quick component tests."""
    print("="*50)
    print("DAY 1 QUICK COMPONENT TEST")
    print("="*50)
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Dueling DQN", test_dueling_dqn),
        ("Prioritized Replay Buffer", test_replay_buffer),
        ("MEC Environment", test_mec_environment)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("QUICK TEST RESULTS:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:.<35} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL COMPONENTS WORKING! Day 1 implementation is ready!")
        print("✓ Dueling DQN architecture implemented")
        print("✓ Prioritized Experience Replay implemented") 
        print("✓ MEC Environment simulation ready")
        print("✓ Configuration system working")
        print("\nReady to proceed to Day 2: SFEA Algorithm Development")
    else:
        print(f"\n⚠️  {total - passed} components need attention")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    print(f"\nQuick test completed. Status: {'SUCCESS' if success else 'FAILED'}")
