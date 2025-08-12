"""
Implementation Validation - Core Logic Test
Tests the implementation logic without heavy ML dependencies
"""

import sys
import os
import yaml
import numpy as np
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def test_configuration_system():
    """Test the configuration loading system."""
    print("Testing Configuration System...")
    try:
        # Test base config
        with open('configs/base_config.yaml', 'r') as f:
            base_config = yaml.safe_load(f)
        
        # Test network config
        with open('configs/network_params.yaml', 'r') as f:
            network_config = yaml.safe_load(f)
        
        # Verify key sections exist
        required_sections = ['learning', 'network', 'dqn', 'environment']
        missing_sections = []
        
        for section in required_sections:
            if section not in base_config:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"✗ Missing config sections: {missing_sections}")
            return False
        
        print(f"✓ Base config loaded: {list(base_config.keys())}")
        print(f"✓ Network config loaded: {list(network_config.keys())}")
        print("✓ All required sections present")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_file_structure_integrity():
    """Test file structure and basic imports."""
    print("Testing File Structure Integrity...")
    try:
        # Test core module structure
        core_files = ['fl_coordinator.py', 'client_manager.py', 'server_aggregator.py']
        for file in core_files:
            if not os.path.exists(f'core/{file}'):
                print(f"✗ Missing core file: {file}")
                return False
        
        # Test agents module structure
        agent_files = ['dueling_dqn.py', 'prioritized_replay.py']
        for file in agent_files:
            if not os.path.exists(f'agents/{file}'):
                print(f"✗ Missing agent file: {file}")
                return False
        
        # Test simulation module structure
        sim_files = ['mec_environment.py', 'ns3_interface.py']
        for file in sim_files:
            if not os.path.exists(f'simulation/{file}'):
                print(f"✗ Missing simulation file: {file}")
                return False
        
        # Test utils module structure
        util_files = ['logger.py', 'file_manager.py']
        for file in util_files:
            if not os.path.exists(f'utils/{file}'):
                print(f"✗ Missing util file: {file}")
                return False
        
        print("✓ All core modules present")
        print("✓ All agent modules present")
        print("✓ All simulation modules present")
        print("✓ All utility modules present")
        return True
        
    except Exception as e:
        print(f"✗ File structure test failed: {e}")
        return False

def test_logger_functionality():
    """Test the logging system."""
    print("Testing Logger Functionality...")
    try:
        from utils.logger import configure_experiment_logging
        
        # Test logger configuration
        experiment_name = "test_experiment"
        config = {
            'learning': {'learning_rate': 0.001},
            'network': {'num_mobile_users': 10}
        }
        
        main_logger, perf_logger, metrics_logger = configure_experiment_logging(
            experiment_name, config, 'logs'
        )
        
        # Test logging
        main_logger.info("Test log message")
        perf_logger.log_timing("test_operation", 1.5)
        metrics_logger.log_training_metrics(1, {'loss': 0.5, 'accuracy': 0.8})
        
        print("✓ Logger configuration successful")
        print("✓ Main logging functional")
        print("✓ Performance logging functional")
        print("✓ Metrics logging functional")
        return True
        
    except Exception as e:
        print(f"✗ Logger test failed: {e}")
        return False

def test_file_manager():
    """Test the file management system."""
    print("Testing File Manager...")
    try:
        from utils.file_manager import FileManager
        
        # Test file manager initialization
        fm = FileManager('test_experiment', 'logs')
        
        # Test directory creation
        test_data = {'test_key': 'test_value', 'metrics': [1, 2, 3]}
        
        # Test save functionality
        fm.save_data(test_data, 'test_data.json')
        print("✓ File manager initialization successful")
        print("✓ Data saving functional")
        
        # Test directory structure
        info = fm.get_directory_info()
        print(f"✓ Directory info: {info}")
        
        return True
        
    except Exception as e:
        print(f"✗ File manager test failed: {e}")
        return False

def test_mec_environment_basic():
    """Test MEC environment basic functionality."""
    print("Testing MEC Environment (Basic)...")
    try:
        # Load configuration
        with open('configs/base_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        from simulation.mec_environment import MECEnvironment
        
        # Create environment
        env = MECEnvironment(config)
        
        # Test basic properties
        print(f"✓ Environment created")
        print(f"✓ Number of edge servers: {env.num_edge_servers}")
        print(f"✓ Area dimensions: {env.area_width}x{env.area_height}")
        
        # Test state generation
        state = env.reset(user_id=0)
        print(f"✓ State generation successful, shape: {state.shape}")
        
        # Test action space
        if hasattr(env, 'action_space_size'):
            print(f"✓ Action space size: {env.action_space_size}")
        
        return True
        
    except Exception as e:
        print(f"✗ MEC Environment test failed: {e}")
        return False

def test_ns3_interface():
    """Test NS-3 interface (mock) functionality."""
    print("Testing NS-3 Interface (Mock)...")
    try:
        # Load configuration  
        with open('configs/base_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        from simulation.ns3_interface import MockNS3Interface
        
        # Create interface
        ns3_interface = MockNS3Interface(config)
        
        # Test basic functionality
        scenario_config = {
            'num_users': 5,
            'num_servers': 3,
            'output_directory': 'test_output'
        }
        
        # Test simulation start
        success = ns3_interface.start_simulation(scenario_config)
        print(f"✓ Simulation start: {success}")
        
        if success:
            # Test status
            status = ns3_interface.get_simulation_status()
            print(f"✓ Simulation status: {status}")
            
            # Test stop
            ns3_interface.stop_simulation()
            print("✓ Simulation stop successful")
        
        return True
        
    except Exception as e:
        print(f"✗ NS-3 Interface test failed: {e}")
        return False

def main():
    """Run implementation validation tests."""
    print("="*60)
    print("IMPLEMENTATION VALIDATION")
    print("Core Logic and Structure Testing")
    print("="*60)
    
    tests = [
        ("Configuration System", test_configuration_system),
        ("File Structure Integrity", test_file_structure_integrity),
        ("Logger Functionality", test_logger_functionality),
        ("File Manager", test_file_manager),
        ("MEC Environment (Basic)", test_mec_environment_basic),
        ("NS-3 Interface (Mock)", test_ns3_interface)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"{test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"{test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
        print("-" * 40)
    
    # Summary
    print("\nVALIDATION RESULTS:")
    print("="*40)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:.<35} {status}")
    
    print("-" * 40)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nimplementation validated!")
        print("- configuration system working")
        print("- file structure complete")
        print("- logging system functional")
        print("- file management operational")
        print("- mec environment basic functions working")
        print("- ns-3 interface (mock) operational")
        print("\nstatus: ready for ml component integration")
        print("next: install ml dependencies and test dqn/fl components")
        print("then: proceed to algorithm development")
    else:
        print(f"\n{total - passed} components need attention")
        print("fix issues: before proceeding to ml component testing")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    print(f"\nValidation completed. Status: {'SUCCESS' if success else 'NEEDS_FIXES'}")
