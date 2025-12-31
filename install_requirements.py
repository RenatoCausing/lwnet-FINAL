#!/usr/bin/env python
"""
Installation script for all required packages.
Automatically detects if CUDA is available and installs appropriate PyTorch version.
"""

import subprocess
import sys
import platform

def run_command(command):
    """Run a shell command and print output"""
    print(f"\n{'='*60}")
    print(f"Running: {command}")
    print('='*60)
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"ERROR: Command failed with return code {result.returncode}")
        return False
    return True

def check_cuda():
    """Check if CUDA is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def main():
    print("="*60)
    print("INSTALLING REQUIRED PACKAGES")
    print("="*60)
    
    # Upgrade pip first
    print("\n[1/3] Upgrading pip...")
    if not run_command(f"{sys.executable} -m pip install --upgrade pip"):
        print("WARNING: Failed to upgrade pip, continuing anyway...")
    
    # Install PyTorch with CUDA support if available
    print("\n[2/3] Installing PyTorch...")
    
    # Check system
    system = platform.system()
    
    # Install PyTorch (adjust based on your needs)
    # For CUDA 11.8 (common for most GPUs)
    cuda_command = f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    
    # For CPU only
    cpu_command = f"{sys.executable} -m pip install torch torchvision torchaudio"
    
    print("\nSelect PyTorch version:")
    print("1. PyTorch with CUDA 11.8 (recommended for NVIDIA GPUs)")
    print("2. PyTorch with CUDA 12.1 (for newer GPUs)")
    print("3. PyTorch CPU only")
    
    choice = input("\nEnter choice (1/2/3) [default: 1]: ").strip()
    
    if choice == "2":
        torch_command = f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    elif choice == "3":
        torch_command = cpu_command
    else:
        torch_command = cuda_command
    
    if not run_command(torch_command):
        print("ERROR: Failed to install PyTorch")
        return False
    
    # Install other requirements
    print("\n[3/3] Installing other required packages...")
    requirements = [
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "Pillow>=8.0.0",
        "scikit-learn>=0.24.0",
        "scipy>=1.6.0",
        "tqdm>=4.60.0",
    ]
    
    for req in requirements:
        if not run_command(f"{sys.executable} -m pip install {req}"):
            print(f"ERROR: Failed to install {req}")
            return False
    
    # Verify installation
    print("\n" + "="*60)
    print("VERIFYING INSTALLATION")
    print("="*60)
    
    try:
        import torch
        import torchvision
        import numpy
        import pandas
        import PIL
        import sklearn
        import scipy
        import tqdm
        
        print("\n✓ All packages installed successfully!")
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"Torchvision version: {torchvision.__version__}")
        print(f"NumPy version: {numpy.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
        
        print("\n" + "="*60)
        print("INSTALLATION COMPLETE!")
        print("="*60)
        return True
        
    except ImportError as e:
        print(f"\n✗ Installation verification failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
