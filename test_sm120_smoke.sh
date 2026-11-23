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
echo "=== Test 2: Inner Hash (Mining Critical) ==="
python3 -m pytest tests/test_inner_hash.py -v -x 2>&1 | tee ../../test_inner_hash.log
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✓ Inner hash PASSED (bit-exact mining hash verified)"
else
    echo "✗ Inner hash FAILED - check test_inner_hash.log"
    echo "CRITICAL: Mining proofs will be invalid without correct inner hash"
    exit 1
fi

echo ""
echo "=== Test 3: Noise Generation ==="
python3 -m pytest tests/test_noise_gen.py -v -x 2>&1 | tee ../../test_noise.log
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✓ Noise generation PASSED"
else
    echo "✗ Noise generation FAILED - check test_noise.log"
    exit 1
fi

echo ""
echo "=== Test 4: GEMM Numerical Correctness ==="
cd ../..
python3 miner/pearl-gemm/tests/test_sm120_correctness.py 2>&1 | tee test_gemm_numerical.log
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✓ GEMM numerical output PASSED"
else
    echo "✗ GEMM numerical FAILED - check test_gemm_numerical.log"
    echo "If error is 'unspecified launch failure', kernel implementation needs debugging"
    exit 1
fi

echo ""
echo "=== WARNING: Numerical tests do NOT validate mining proofs ==="
echo "These tests check GEMM output, NOT:"
echo "  - Inner hash / transcript extraction"
echo "  - Commitment hashes (Merkle + V3 salting)"
echo "  - Proof generation"
echo ""
echo "Full mining validation requires:"
echo "  - pytest miner/pearl-gemm/tests/test_pearl_gemm.py::TestGemm::test_inner_hash_count -v"
echo "  - Actual simnet share acceptance"
echo ""
echo "=== Basic Smoke Tests PASSED ==="
echo "Next: Run full mining tests with skip_reduction=False"
