#!/usr/bin/env python3
"""
GPU Diagnostic Script
Check for GPU/CUDA availability and configuration
"""

import sys
import os
import subprocess

def run_command(cmd):
    """Run a command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", str(e), 1

def check_nvidia_driver():
    """Check NVIDIA driver installation."""
    print("NVIDIA Driver Check")
    stdout, stderr, code = run_command("nvidia-smi")
    if code == 0:
        print("✓ nvidia-smi found:")
        print(stdout)
        return True
    else:
        print("✗ nvidia-smi not found or failed:")
        print(f"Error: {stderr}")
        return False

def check_cuda_toolkit():
    """Check CUDA toolkit installation."""
    print("CUDA Toolkit Check")
    stdout, stderr, code = run_command("nvcc --version")
    if code == 0:
        print("✓ CUDA compiler found:")
        print(stdout)
        return True
    else:
        print("✗ CUDA compiler (nvcc) not found:")
        print(f"Error: {stderr}")
        return False

def check_pytorch():
    """Check PyTorch installation and CUDA support."""
    print("PyTorch Check")
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        # Check CUDA compilation
        if torch.version.cuda:
            print(f"✓ PyTorch compiled with CUDA: {torch.version.cuda}")
        else:
            print("✗ PyTorch NOT compiled with CUDA")
            return False
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print("✓ CUDA is available in PyTorch")
            print(f"✓ CUDA device count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"✓ GPU {i}: {torch.cuda.get_device_name(i)}")
                
            # Test GPU computation
            try:
                device = torch.device('cuda:0')
                x = torch.randn(100, 100, device=device)
                y = torch.randn(100, 100, device=device)
                z = torch.mm(x, y)
                print("✓ GPU computation test passed")
                return True
            except Exception as e:
                print(f"✗ GPU computation test failed: {e}")
                return False
        else:
            print("✗ CUDA is NOT available in PyTorch")
            return False
            
    except ImportError:
        print("✗ PyTorch not installed")
        return False
    except Exception as e:
        print(f"✗ PyTorch check failed: {e}")
        return False

def check_environment():
    """Check environment variables."""
    print("Environment Variables")
    
    cuda_vars = ['CUDA_HOME', 'CUDA_PATH', 'CUDA_TOOLKIT_ROOT_DIR', 'LD_LIBRARY_PATH']
    
    for var in cuda_vars:
        value = os.environ.get(var)
        if value:
            print(f"✓ {var}: {value}")
        else:
            print(f"✗ {var}: Not set")

def check_wsl_gpu():
    """Check WSL GPU support specifically."""
    print("WSL GPU Support Check")
    
    # Check if running in WSL
    try:
        with open('/proc/version', 'r') as f:
            version_info = f.read()
            if 'microsoft' in version_info.lower() or 'wsl' in version_info.lower():
                print("✓ Running in WSL")
                
                # Check for WSL GPU support
                stdout, stderr, code = run_command("ls /usr/lib/wsl/lib/")
                if 'nvidia' in stdout.lower():
                    print("✓ WSL NVIDIA libraries found")
                else:
                    print("✗ WSL NVIDIA libraries not found")
                    print("Install WSL CUDA support from NVIDIA")
            else:
                print("✓ Running in native Linux (not WSL)")
                
    except Exception as e:
        print(f"Could not determine WSL status: {e}")

def provide_recommendations():
    """Provide installation recommendations."""
    print("Recommendations")
    
    print("If GPU is not working, try these steps:")
    print("\n1. Install NVIDIA drivers (Windows):")
    print("   - Download from https://www.nvidia.com/drivers")
    print("   - Install latest Game Ready or Studio drivers")
    
    print("\n2. Install WSL CUDA support:")
    print("   - Download WSL CUDA toolkit from NVIDIA")
    print("   - Follow: https://docs.nvidia.com/cuda/wsl-user-guide/")
    
    print("\n3. Install PyTorch with CUDA:")
    print("   pip uninstall torch torchvision torchaudio")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
    print("\n4. Verify installation:")
    print("   python -c \"import torch; print(torch.cuda.is_available())\"")

def main():
    """Main diagnostic function."""
    print("GPU/CUDA Diagnostic Tool")
    print("="*50)
    
    results = []
    
    # Run all checks
    results.append(("NVIDIA Driver", check_nvidia_driver()))
    results.append(("CUDA Toolkit", check_cuda_toolkit()))
    results.append(("PyTorch CUDA", check_pytorch()))
    
    check_environment()
    check_wsl_gpu()
    
    # Summary
    print("SUMMARY")
    all_good = True
    for check_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{check_name}: {status}")
        if not result:
            all_good = False
    
    if all_good:
        print("All GPU checks passed! Your system is ready for GPU acceleration.")
    else:
        print("Some GPU checks failed. See recommendations below.")
        provide_recommendations()

if __name__ == "__main__":
    main()
