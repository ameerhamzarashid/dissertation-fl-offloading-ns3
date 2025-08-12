"""
Simple test to verify basic Python functionality and imports
"""

print("Starting basic Python test...")

try:
    import sys
    print(f"Python version: {sys.version}")
    
    import os
    print(f"Current directory: {os.getcwd()}")
    
    # Test basic imports that should be available
    import random
    import json
    import time
    from datetime import datetime
    
    print("Basic imports successful")
    
    # Test if torch is available
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print("PyTorch available")
    except ImportError:
        print("PyTorch not available - need to install")
    
    # Test if numpy is available
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
        print("NumPy available")
    except ImportError:
        print("NumPy not available - need to install")
    
    print("\nBasic test completed successfully!")
    
except Exception as e:
    print(f"Error during basic test: {e}")
    import traceback
    traceback.print_exc()
