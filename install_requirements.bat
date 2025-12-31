@echo off
REM Windows batch script to install all requirements

echo ============================================================
echo INSTALLING REQUIRED PACKAGES
echo ============================================================

REM Upgrade pip
echo.
echo [1/3] Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch with CUDA 11.8 (most common)
echo.
echo [2/3] Installing PyTorch with CUDA 11.8...
echo.
echo Select PyTorch version:
echo 1. PyTorch with CUDA 11.8 (recommended for NVIDIA GPUs)
echo 2. PyTorch with CUDA 12.1 (for newer GPUs)
echo 3. PyTorch CPU only
echo.
set /p choice="Enter choice (1/2/3) [default: 1]: "

if "%choice%"=="2" (
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else if "%choice%"=="3" (
    python -m pip install torch torchvision torchaudio
) else (
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
)

REM Install other requirements
echo.
echo [3/3] Installing other required packages...
python -m pip install numpy>=1.19.0 pandas>=1.2.0 Pillow>=8.0.0 scikit-learn>=0.24.0 scipy>=1.6.0 tqdm>=4.60.0

echo.
echo ============================================================
echo INSTALLATION COMPLETE!
echo ============================================================
echo.
echo Run this to verify: python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
echo.
pause
