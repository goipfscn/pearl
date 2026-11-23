#!/bin/bash
# Minimal sm_120 smoke test for RTX PRO 5000
# Run after build_sm120.sh succeeds

set -e

echo "=== Pearl sm_120 Smoke Test - $(date) ==="

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found"
    exit 1
fi

echo "GPU Info:"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

GPU_CC=$(python3 -c "import torch; print(f'{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}')" 2>/dev/null || echo "unknown")
echo "Detected compute capability: $GPU_CC"

if [[ "$GPU_CC" != "120" && "$GPU_CC" != "121" ]]; then
    echo "WARNING: Expected sm_120/121, got sm_$GPU_CC"
fi

echo ""
echo "=== Test 1: Hash Kernels (Simple) ==="
cd miner/pearl-gemm
python3 -m pytest tests/test_tensor_hash.py -v -x 2>&1 | tee ../../test_hash.log
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✓ Hash kernels PASSED"
else
    echo "✗ Hash kernels FAILED - check test_hash.log"
    exit 1
fi

echo ""
echo "=== Test 2: Noise Generation ==="
python3 -m pytest tests/test_noise_gen.py -v -x 2>&1 | tee ../../test_noise.log
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✓ Noise generation PASSED"
else
    echo "✗ Noise generation FAILED - check test_noise.log"
    exit 1
fi

echo ""
echo "=== Test 3: GEMM Correctness ==="
cd ../..
python3 miner/pearl-gemm/tests/test_sm120_correctness.py 2>&1 | tee test_gemm.log
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✓ GEMM correctness PASSED"
else
    echo "✗ GEMM correctness FAILED - check test_gemm.log"
    echo "If error is 'unspecified launch failure', kernel implementation needs debugging"
    exit 1
fi

echo ""
echo "=== All Smoke Tests PASSED ==="
echo "Next: Test mining integration on simnet (see test_mining_simnet.sh)"
