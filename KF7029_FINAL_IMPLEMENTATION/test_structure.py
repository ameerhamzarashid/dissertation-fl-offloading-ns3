"""
Day 1 Implementation Structure Test
Tests the baseline implementation structure without heavy dependencies
"""

import os
import sys
import importlib.util
from pathlib import Path

def test_file_structure():
    """Test if all required files are present."""
    print("Testing file structure...")
    
    required_files = [
        'core/__init__.py',
        'core/fl_coordinator.py',
        'core/client_manager.py', 
        'core/server_aggregator.py',
        'agents/__init__.py',
        'agents/dueling_dqn.py',
        'agents/prioritized_replay.py',
        'simulation/__init__.py',
        'simulation/mec_environment.py',
        'simulation/ns3_interface.py',
        'utils/__init__.py',
        'utils/logger.py',
        'utils/file_manager.py',
        'configs/base_config.yaml',
        'configs/network_params.yaml'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"Missing files: {missing_files}")
        return False
    else:
        print("All required files are present!")
        return True


def test_python_syntax():
    """Test if all Python files have valid syntax."""
    print("Testing Python syntax...")
    
    python_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py') and file != 'test_structure.py':
                python_files.append(os.path.join(root, file))
    
    syntax_errors = []
    for file_path in python_files:
        try:
            with open(file_path, 'r') as f:
                compile(f.read(), file_path, 'exec')
        except SyntaxError as e:
            syntax_errors.append((file_path, str(e)))
        except Exception as e:
            syntax_errors.append((file_path, f"Other error: {str(e)}"))
    
    if syntax_errors:
        print("Syntax errors found:")
        for file_path, error in syntax_errors:
            print(f"  {file_path}: {error}")
        return False
    else:
        print(f"All {len(python_files)} Python files have valid syntax!")
        return True


def test_imports_structure():
    """Test if imports are properly structured (syntax only)."""
    print("Testing import structure...")
    
    # Test key files for import structure
    key_files = [
        'core/fl_coordinator.py',
        'agents/dueling_dqn.py', 
        'agents/prioritized_replay.py',
        'simulation/mec_environment.py'
    ]
    
    import_issues = []
    for file_path in key_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                # Check for relative imports
                if 'from .' in content or 'from ..' in content:
                    print(f"  {file_path}: Uses relative imports ✓")
                
                # Check for absolute imports
                if 'from core.' in content or 'from agents.' in content:
                    print(f"  {file_path}: Uses absolute imports ✓")
                    
            except Exception as e:
                import_issues.append((file_path, str(e)))
    
    if import_issues:
        print("Import structure issues:")
        for file_path, error in import_issues:
            print(f"  {file_path}: {error}")
        return False
    else:
        print("Import structure looks good!")
        return True


def test_yaml_configs():
    """Test if YAML configuration files are valid."""
    print("Testing YAML configurations...")
    
    yaml_files = [
        'configs/base_config.yaml',
        'configs/network_params.yaml'
    ]
    
    yaml_errors = []
    for file_path in yaml_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    # Basic YAML syntax check
                    if ':' in content and not content.strip().startswith('#'):
                        print(f"  {file_path}: Valid YAML structure ✓")
                    else:
                        yaml_errors.append((file_path, "No valid YAML structure found"))
            except Exception as e:
                yaml_errors.append((file_path, str(e)))
        else:
            yaml_errors.append((file_path, "File not found"))
    
    if yaml_errors:
        print("YAML configuration issues:")
        for file_path, error in yaml_errors:
            print(f"  {file_path}: {error}")
        return False
    else:
        print("YAML configurations are valid!")
        return True


def test_directory_structure():
    """Test directory structure."""
    print("Testing directory structure...")
    
    required_dirs = ['core', 'agents', 'simulation', 'utils', 'configs', 'logs']
    missing_dirs = []
    
    for dir_name in required_dirs:
        if not os.path.isdir(dir_name):
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"Missing directories: {missing_dirs}")
        return False
    else:
        print("All required directories are present!")
        return True


def main():
    """Main function to run structure tests."""
    print("="*60)
    print("DAY 1 IMPLEMENTATION STRUCTURE TEST")
    print("Federated Learning with Dueling DQN + PER + MEC Environment")
    print("="*60)
    
    # Change to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    test_results = []
    
    # Run structure tests
    tests = [
        ("Directory Structure", test_directory_structure),
        ("File Structure", test_file_structure),
        ("Python Syntax", test_python_syntax),
        ("Import Structure", test_imports_structure),
        ("YAML Configurations", test_yaml_configs)
    ]
    
    for test_name, test_function in tests:
        try:
            print(f"\nRunning {test_name} test...")
            result = test_function()
            test_results.append((test_name, result))
            print(f"{test_name} test: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"{test_name} test FAILED with error: {e}")
            test_results.append((test_name, False))
        print("-" * 40)
    
    # Summary
    print("\nDAY 1 STRUCTURE TEST RESULTS:")
    print("="*40)
    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:.<30} {status}")
    
    print("-" * 40)
    print(f"Total: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\nSTRUCTURE TESTS PASSED! Implementation structure is ready.")
        print("Next: Install dependencies and run full functionality tests")
    else:
        print(f"\n{total_tests - passed_tests} tests failed. Please fix structure issues.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    print(f"\nStructure test completed. Status: {'SUCCESS' if success else 'FAILED'}")
