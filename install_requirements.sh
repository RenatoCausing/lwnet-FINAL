#!/bin/bash
# Linux/Mac shell script to install all requirements

echo "============================================================"
echo "INSTALLING REQUIRED PACKAGES"
echo "============================================================"

# Upgrade pip
echo ""
echo "[1/3] Upgrading pip..."
python -m pip install --upgrade pip

# Install PyTorch
echo ""
echo "[2/3] Installing PyTorch..."
echo ""
echo "Select PyTorch version:"
echo "1. PyTorch with CUDA 11.8 (recommended for NVIDIA GPUs)"
echo "2. PyTorch with CUDA 12.1 (for newer GPUs)"
echo "3. PyTorch CPU only"
echo ""
read -p "Enter choice (1/2/3) [default: 1]: " choice

if [ "$choice" == "2" ]; then
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
elif [ "$choice" == "3" ]; then
    python -m pip install torch torchvision torchaudio
else
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

# Install other requirements
echo ""
echo "[3/3] Installing other required packages..."
python -m pip install "numpy>=1.19.0" "pandas>=1.2.0" "Pillow>=8.0.0" "scikit-learn>=0.24.0" "scipy>=1.6.0" "tqdm>=4.60.0"

echo ""
echo "============================================================"
echo "INSTALLATION COMPLETE!"
echo "============================================================"
echo ""
echo "Run this to verify: python -c \"import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())\""
echo ""
