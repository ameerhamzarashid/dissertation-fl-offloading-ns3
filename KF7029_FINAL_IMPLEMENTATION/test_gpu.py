"""
GPU Test Script
Quick test to verify GPU integration and performance
"""

import torch
import numpy as np
import time
from utils.gpu_utils import print_gpu_summary, setup_device, monitor_gpu_usage

def test_gpu_performance():
    """Test GPU performance vs CPU."""
    print("testing gpu performance...")
    
    # Setup device
    device = setup_device(preferred_device='cuda')
    print(f"using device: {device}")
    
    # Create test tensors
    size = 1000
    a = torch.randn(size, size)
    b = torch.randn(size, size)
    
    # CPU test
    start_time = time.time()
    c_cpu = torch.mm(a, b)
    cpu_time = time.time() - start_time
    print(f"cpu matrix multiplication time: {cpu_time:.4f} seconds")
    
    # GPU test (if available)
    if device.type == 'cuda':
        a_gpu = a.to(device)
        b_gpu = b.to(device)
        
        # Warm up GPU
        torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        
        start_time = time.time()
        c_gpu = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        print(f"gpu matrix multiplication time: {gpu_time:.4f} seconds")
        speedup = cpu_time / gpu_time
        print(f"gpu speedup: {speedup:.2f}x faster")
        
        # Monitor GPU usage
        gpu_usage = monitor_gpu_usage()
        if gpu_usage.get('gpu_available'):
            print(f"gpu memory used: {gpu_usage['allocated_mb']:.1f} mb")
            print(f"gpu utilization: {gpu_usage['utilization_percent']:.1f}%")
    else:
        print("gpu not available, running on cpu only")

def test_neural_network():
    """Test neural network on GPU."""
    print("\ntesting neural network on gpu...")
    
    device = setup_device(preferred_device='cuda')
    
    # Simple neural network
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 10)
    ).to(device)
    
    print(f"model device: {next(model.parameters()).device}")
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 100).to(device)
    
    start_time = time.time()
    output = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    inference_time = time.time() - start_time
    
    print(f"inference time: {inference_time:.4f} seconds")
    print(f"output shape: {output.shape}")
    print(f"output device: {output.device}")

if __name__ == "__main__":
    print("="*50)
    print("gpu integration test")
    print("="*50)
    
    # Print GPU summary
    print_gpu_summary()
    
    # Run performance tests
    test_gpu_performance()
    test_neural_network()
    
    print("\ngpu test completed!")
