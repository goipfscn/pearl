# Pearl sm_120 Blackwell Build Guide

Complete instructions for building and testing Pearl INT7 miner on Blackwell RTX PRO 5000 / RTX 50-series GPUs.

## Prerequisites

### System Requirements
- **GPU**: NVIDIA RTX PRO 5000 72GB, RTX 5090/5080/5070 (Blackwell sm_120/121)
- **Driver**: NVIDIA driver 560+ (supports Blackwell)
- **CUDA**: CUDA Toolkit 12.4+ or 13.0+ 
- **OS**: Linux (Ubuntu 22.04+ recommended), WSL2 also works
- **RAM**: 32GB+ recommended
- **Disk**: 50GB free space

### Software Dependencies
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager (0.5.0+)
- Git
- Build essentials: `sudo apt install build-essential`

## Part 1: Build pearl-gemm for sm_120

### Step 1.1: Clone Repository

```bash
# Clone your fork
git clone https://github.com/goipfscn/pearl.git
cd pearl

# Switch to sm_120 branch
git checkout cursor/sm120-blackwell-support-1064
```

### Step 1.2: Configure Build for sm_120

```bash
# Set target architecture
export PEARL_GEMM_ARCH=sm_120a

# Force rebuild (skip prebuilt wheels)
export PEARL_GEMM_FORCE_BUILD=1

# Optional: control parallelism (default auto-detects)
export MAX_JOBS=8

# Verify CUDA
nvcc --version
nvidia-smi
```

**Expected output from `nvidia-smi`**:
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.xx.xx              Driver Version: 560.xx.xx      CUDA Version: 12.6    |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX PRO 5000 Ada ...  Off |   00000000:01:00.0  On |                  N/A |
| 30%   35C    P8             25W /  260W |     512MiB /  73728MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

### Step 1.3: Build pearl-gemm

```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Build pearl-gemm (this takes 10-20 minutes)
cd /workspace  # or your pearl directory
uv sync --package pearl-gemm --reinstall-package pearl-gemm

# Monitor build progress
# - Compilation generates ~8-12 .cu instantiation files
# - Each .cu -> .o takes 2-5 minutes (parallel)
# - Final link creates pearl_gemm_cuda.cpython-*.so
```

**Build output should show**:
```
target architecture = sm_120a (arch=compute_120a,code=sm_120a)
...
Writing template instantiations to .../csrc/gemm/instantiations
Writing static switches to .../csrc/gemm
...
building 'pearl_gemm_cuda' extension
...
creating build/lib.linux-x86_64-cpython-312
...
Successfully installed pearl-gemm-0.1.0
```

### Step 1.4: Verify Build

```bash
# Check generated shared library
find . -name "pearl_gemm_cuda*.so" -exec ls -lh {} \;

# Verify architecture
cuobjdump --list-elf $(find . -name "pearl_gemm_cuda*.so" | head -1) | grep arch

# Expected: Multiple sm_120 cubins
# Example output:
# arch = sm_120a
# arch = sm_120a
# ...
```

```bash
# Test Python import
uv run python -c "
import torch
import pearl_gemm
import pearl_gemm_cuda
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'Device: {torch.cuda.get_device_name(0)}')
print(f'Compute: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}')
print('pearl_gemm import: OK')
"

# Expected output:
# PyTorch: 2.5.1+cu124 (or similar)
# CUDA: 12.4 (or 13.0+)
# Device: NVIDIA RTX PRO 5000 Ada Generation
# Compute: sm_120
# pearl_gemm import: OK
```

### Common Build Issues

**Issue**: `CUDA_HOME not set`
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

**Issue**: `nvcc: command not found`
```bash
# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-4
```

**Issue**: Out of memory during build
```bash
# Reduce parallelism
export MAX_JOBS=4  # or 2 for systems with <32GB RAM
```

**Issue**: `torch._C._GLIBCXX_USE_CXX11_ABI` mismatch
```bash
# Force CXX11 ABI (for NVIDIA containers)
export PEARL_GEMM_FORCE_CXX11_ABI=1
```

## Part 2: Build Full Pearl Stack

### Step 2.1: Build Blockchain Components

```bash
cd /workspace
task build:blockchain
# This builds: pearld, prlctl, oyster, oystercli
# Binaries appear in bin/
```

### Step 2.2: Build Remaining Miner Components

```bash
# Build all miner Python packages
task build:miner

# Or individually:
uv sync --package miner-utils
uv sync --package miner-base
uv sync --package pearl-gateway
uv sync --package vllm-miner
```

### Step 2.3: Create Wallet

```bash
./bin/oyster -u rpcuser -P rpcpass --create
# Follow prompts, save seed phrase

# Start wallet daemon
./bin/oyster -u rpcuser -P rpcpass &

# Generate mining address
./bin/prlctl -u rpcuser -P rpcpass -s https://localhost:44207 getnewaddress
# Save this address for mining
```

## Part 3: Docker Build (Alternative)

### Step 3.1: Build Docker Image

```bash
cd /workspace

# Create Dockerfile with sm_120 build
cat > miner/vllm-miner/Dockerfile.sm120 <<'EOF'
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3-pip \
    git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Copy repository
WORKDIR /work
COPY . /work/

# Build for sm_120a
ENV PEARL_GEMM_ARCH=sm_120a
ENV PEARL_GEMM_FORCE_BUILD=1
ENV MAX_JOBS=8

RUN uv sync --package vllm-miner

ENTRYPOINT ["uv", "run", "vllm", "serve"]
EOF

# Build image
docker build -t pearl-miner-sm120:latest \
  -f miner/vllm-miner/Dockerfile.sm120 .
```

### Step 3.2: Run Docker Container

```bash
docker run --rm -it --gpus all \
  -p 8000:8000 -p 8337:8337 \
  -e PEARLD_RPC_URL=http://host.docker.internal:18556 \
  -e PEARLD_RPC_USER=rpcuser \
  -e PEARLD_RPC_PASSWORD=rpcpass \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --shm-size 16g \
  pearl-miner-sm120:latest \
  pearl-ai/Llama-3.3-8B-Instruct-pearl \
  --host 0.0.0.0 --port 8000 --max-model-len 4096
```

## Part 4: Isolated venv Build (No uv)

### Step 4.1: Create venv

```bash
python3.12 -m venv ~/pearl-sm120-env
source ~/pearl-sm120-env/bin/activate

# Install build dependencies
pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Step 4.2: Build pearl-gemm

```bash
cd /workspace/miner/pearl-gemm

export PEARL_GEMM_ARCH=sm_120a
export PEARL_GEMM_FORCE_BUILD=1

pip install -e .
```

### Step 4.3: Build Dependencies

```bash
cd /workspace

# Install in dependency order
pip install -e miner/miner-utils
pip install -e miner/miner-base
pip install -e miner/pearl-gateway
pip install -e miner/vllm-miner
```

## Part 5: Verification

### CPU Tests (No GPU Mining)

```bash
# Hash kernels (simpler, should work)
uv run pytest miner/pearl-gemm/tests/test_tensor_hash.py -v
uv run pytest miner/pearl-gemm/tests/test_inner_hash.py -v
uv run pytest miner/pearl-gemm/tests/test_noise_gen.py -v
```

### GPU Tests (Requires Implementation Complete)

```bash
# sm_120 correctness suite
uv run python miner/pearl-gemm/tests/test_sm120_correctness.py

# Full GEMM test suite
uv run pytest miner/pearl-gemm/tests/test_pearl_gemm.py -v \
  --ignore=test_vllm_execution.py
```

## Part 6: Rollback / Cleanup

### Remove sm_120 Build

```bash
# Remove compiled artifacts
cd /workspace
rm -rf miner/pearl-gemm/build
rm -rf miner/pearl-gemm/src/pearl_gemm_cuda*.so
rm -rf .venv

# Rebuild for Hopper (if needed)
unset PEARL_GEMM_ARCH
export PEARL_GEMM_FORCE_BUILD=1
uv sync --package pearl-gemm --reinstall-package pearl-gemm
```

### Save Build Artifacts

```bash
# Archive for transfer
cd /workspace
tar czf pearl-sm120-build-$(date +%Y%m%d).tar.gz \
  miner/pearl-gemm/build \
  miner/pearl-gemm/src/*.so \
  miner/pearl-gemm/csrc/gemm/instantiations \
  miner/pearl-gemm/csrc/gemm/static_switch*.h

# Save environment info
uv run python -c "
import torch
import sys
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'cuDNN: {torch.backends.cudnn.version()}')
print(f'Device: {torch.cuda.get_device_name(0)}')
print(f'Compute: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}')
print(f'CUDA home: {torch.utils.cpp_extension.CUDA_HOME}')
" > pearl-sm120-env.txt
```

## Troubleshooting

### GPU Not Detected

```bash
# Check driver
nvidia-smi

# Check CUDA devices
python3 -c "import torch; print(torch.cuda.is_available())"

# If false, reinstall PyTorch with correct CUDA version
pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu124
```

### Build Hangs

- Check RAM usage: `free -h`
- Reduce `MAX_JOBS` if memory is low
- Kill zombie processes: `pkill -9 nvcc`
- Clear build cache: `rm -rf build/ src/pearl_gemm_cuda*.so`

### Runtime "unspecified launch failure"

This indicates the kernel modifications are incomplete. The hash kernels should work, but GEMM kernels will fail until collective_mainloop.hpp and collective_epilogue.hpp changes are applied from PR #221.

**Current Status**: Build succeeds, hash kernels work, GEMM kernels need completion.

**Next**: Follow SM120_GPU_TEST_GUIDE.md for hardware validation protocol.
