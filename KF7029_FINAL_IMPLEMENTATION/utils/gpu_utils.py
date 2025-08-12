"""
GPU Utilities for PyTorch
Provides GPU detection, configuration, and optimization utilities.
"""

import torch
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def get_device_info() -> Dict[str, Any]:
    """
    Get information about available computing devices.
    
    Returns:
        Dict containing device information including GPU details.
    """
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': 0,
        'current_device': None,
        'gpu_names': [],
        'memory_info': {},
        'recommended_device': 'cpu'
    }
    
    if torch.cuda.is_available():
        device_info['device_count'] = torch.cuda.device_count()
        device_info['current_device'] = torch.cuda.current_device()
        
        for i in range(device_info['device_count']):
            gpu_name = torch.cuda.get_device_name(i)
            device_info['gpu_names'].append(gpu_name)
            
            # Get memory information
            if i == device_info['current_device']:
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_reserved = torch.cuda.memory_reserved(i)
                max_memory = torch.cuda.max_memory_allocated(i)
                
                device_info['memory_info'] = {
                    'allocated_mb': memory_allocated / (1024**2),
                    'reserved_mb': memory_reserved / (1024**2),
                    'max_allocated_mb': max_memory / (1024**2)
                }
        
        device_info['recommended_device'] = 'cuda'
        logger.info(f"found {device_info['device_count']} gpu(s): {device_info['gpu_names']}")
    else:
        logger.warning("cuda not available, using cpu")
    
    return device_info

def setup_device(preferred_device: Optional[str] = None, memory_fraction: float = 0.8) -> torch.device:
    """
    Setup and configure the computing device with optimal settings.
    
    Args:
        preferred_device: Preferred device ('cuda', 'cpu', or specific like 'cuda:0')
        memory_fraction: Fraction of GPU memory to allocate (0.1-1.0)
        
    Returns:
        Configured torch.device object
    """
    device_info = get_device_info()
    
    # Determine device
    if preferred_device:
        if preferred_device.startswith('cuda') and not torch.cuda.is_available():
            logger.warning(f"requested {preferred_device} but cuda not available, falling back to cpu")
            device = torch.device('cpu')
        else:
            device = torch.device(preferred_device)
    else:
        device = torch.device(device_info['recommended_device'])
    
    # Configure GPU if using CUDA
    if device.type == 'cuda':
        try:
            # Set memory allocation strategy
            torch.cuda.empty_cache()
            
            # Configure memory growth
            if hasattr(torch.cuda, 'set_memory_fraction'):
                torch.cuda.set_memory_fraction(memory_fraction)
            
            # Optimize for performance
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            logger.info(f"configured gpu device: {device}")
            logger.info(f"gpu name: {torch.cuda.get_device_name(device)}")
            logger.info(f"memory fraction: {memory_fraction}")
            
        except Exception as e:
            logger.error(f"failed to configure gpu: {e}")
            device = torch.device('cpu')
            logger.info("falling back to cpu")
    
    logger.info(f"using device: {device}")
    return device

def optimize_model_for_gpu(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """
    Optimize a PyTorch model for GPU execution.
    
    Args:
        model: PyTorch model to optimize
        device: Target device
        
    Returns:
        Model ready for training
    """
    model = model.to(device)
    
    if device.type == 'cuda':
        # Enable mixed precision training if supported
        try:
            model = model.half() if hasattr(model, 'half') else model
            logger.info("enabled mixed precision training")
        except:
            logger.info("mixed precision not supported, using float32")
        
        # Compile model for better performance (PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile'):
                model = torch.compile(model)
                logger.info("model compiled for optimization")
        except:
            logger.info("model compilation not available")
    
    return model

def monitor_gpu_usage() -> Dict[str, float]:
    """
    Monitor current GPU usage and memory consumption.
    
    Returns:
        Dictionary with GPU usage statistics
    """
    if not torch.cuda.is_available():
        return {'gpu_available': False}
    
    try:
        current_device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(current_device)
        reserved = torch.cuda.memory_reserved(current_device)
        max_allocated = torch.cuda.max_memory_allocated(current_device)
        
        # Get total memory (approximate)
        total_memory = torch.cuda.get_device_properties(current_device).total_memory
        
        return {
            'gpu_available': True,
            'device_id': current_device,
            'allocated_mb': allocated / (1024**2),
            'reserved_mb': reserved / (1024**2),
            'max_allocated_mb': max_allocated / (1024**2),
            'total_memory_mb': total_memory / (1024**2),
            'utilization_percent': (allocated / total_memory) * 100
        }
    except Exception as e:
        logger.error(f"failed to get gpu usage: {e}")
        return {'gpu_available': False, 'error': str(e)}

def clear_gpu_cache():
    """Clear GPU cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("gpu cache cleared")

def get_optimal_batch_size(model: torch.nn.Module, input_shape: tuple, device: torch.device, 
                          max_memory_mb: float = 8000) -> int:
    """
    Estimate optimal batch size for given model and memory constraints.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (excluding batch dimension)
        device: Target device
        max_memory_mb: Maximum memory to use in MB
        
    Returns:
        Recommended batch size
    """
    if device.type != 'cuda':
        return 32  # Conservative batch size for CPU
    
    try:
        model.eval()
        batch_size = 1
        
        # Test with increasing batch sizes
        with torch.no_grad():
            while batch_size <= 512:
                try:
                    test_input = torch.randn(batch_size, *input_shape).to(device)
                    _ = model(test_input)
                    
                    # Check memory usage
                    memory_used = torch.cuda.memory_allocated() / (1024**2)
                    if memory_used > max_memory_mb * 0.8:  # Use 80% of max memory
                        break
                    
                    batch_size *= 2
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        break
                    raise e
        
        optimal_batch_size = max(1, batch_size // 2)
        logger.info(f"estimated optimal batch size: {optimal_batch_size}")
        return optimal_batch_size
        
    except Exception as e:
        logger.error(f"failed to estimate batch size: {e}")
        return 16  # Safe default

def setup_mixed_precision() -> tuple:
    """
    Setup mixed precision training components.
    
    Returns:
        Tuple of (scaler, autocast_context) for mixed precision training
    """
    if torch.cuda.is_available():
        try:
            from torch.cuda.amp import GradScaler, autocast
            scaler = GradScaler()
            logger.info("mixed precision training enabled")
            return scaler, autocast
        except ImportError:
            logger.warning("mixed precision not available in this pytorch version")
    
    return None, None

# Global device configuration
_global_device = None

def get_global_device() -> torch.device:
    """Get the globally configured device."""
    global _global_device
    if _global_device is None:
        _global_device = setup_device()
    return _global_device

def set_global_device(device: torch.device):
    """Set the global device configuration."""
    global _global_device
    _global_device = device
    logger.info(f"global device set to: {device}")

def print_gpu_summary():
    """Print a GPU summary."""
    device_info = get_device_info()
    usage = monitor_gpu_usage()
    
    print("\n" + "="*50)
    print("gpu system summary")
    print("="*50)
    
    if device_info['cuda_available']:
        print(f"cuda available: yes")
        print(f"number of gpus: {device_info['device_count']}")
        print(f"current device: {device_info['current_device']}")
        
        for i, name in enumerate(device_info['gpu_names']):
            print(f"gpu {i}: {name}")
        
        if usage['gpu_available']:
            print(f"\ncurrent memory usage:")
            print(f"  allocated: {usage['allocated_mb']:.1f} mb")
            print(f"  reserved: {usage['reserved_mb']:.1f} mb")
            print(f"  utilization: {usage['utilization_percent']:.1f}%")
    else:
        print("cuda available: no")
        print("using cpu for computation")
    
    print("="*50)
