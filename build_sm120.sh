#!/bin/bash
# Minimal sm_120 build script for RTX PRO 5000
# Run on target machine with CUDA 12.4+ installed

set -e  # Exit on any error

echo "=== Pearl sm_120 Build - $(date) ==="

# Check environment
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Install CUDA toolkit 12.4+ first."
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found."
    exit 1
fi

echo "CUDA version:"
nvcc --version | grep release

echo "Python version:"
python3 --version

# Set architecture
export PEARL_GEMM_ARCH=sm_120a
export PEARL_GEMM_FORCE_BUILD=1
export MAX_JOBS=${MAX_JOBS:-8}

echo "Building for: $PEARL_GEMM_ARCH"
echo "Max parallel jobs: $MAX_JOBS"

# Install uv if needed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Build
echo "Starting build (this takes 10-20 minutes)..."
uv sync --package pearl-gemm --reinstall-package pearl-gemm 2>&1 | tee build_sm120.log

# Verify
echo ""
echo "=== Build Verification ==="

SO_FILE=$(find . -name "pearl_gemm_cuda*.so" 2>/dev/null | head -1)
if [ -z "$SO_FILE" ]; then
    echo "ERROR: No .so file found after build"
    exit 1
fi

echo "Found: $SO_FILE"
ls -lh "$SO_FILE"

echo ""
echo "Checking architecture:"
if command -v cuobjdump &> /dev/null; then
    cuobjdump --list-elf "$SO_FILE" | grep "arch = " | head -5
else
    echo "WARNING: cuobjdump not found, skipping arch verification"
fi

echo ""
echo "Testing import:"
python3 -c "import sys; sys.path.insert(0, 'miner/pearl-gemm/src'); import pearl_gemm; print('pearl_gemm: OK')"
python3 -c "import pearl_gemm_cuda; print('pearl_gemm_cuda: OK')" 2>&1 || echo "Import test: check if in correct directory"

echo ""
echo "=== Build Complete ==="
echo "Log saved to: build_sm120.log"
echo "Next: Run ./test_sm120_smoke.sh"
