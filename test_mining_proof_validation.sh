#!/bin/bash
# Mining proof validation - tests ACTUAL mining path with inner hash
# Run after test_sm120_smoke.sh passes

set -e

echo "=== Pearl sm_120 Mining Proof Validation - $(date) ==="

if ! command -v pytest &> /dev/null; then
    echo "ERROR: pytest not found"
    exit 1
fi

cd miner/pearl-gemm

echo ""
echo "=== Test 1: Inner Hash Count (Mining Hash) ==="
echo "This tests the ACTUAL transcript extraction / xor_reduction path"
python3 -m pytest tests/test_pearl_gemm.py::TestGemm::test_inner_hash_count -v 2>&1 | tee ../../test_inner_hash_count.log
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✓ Inner hash count PASSED - mining hash extraction works"
else
    echo "✗ Inner hash count FAILED"
    echo "CRITICAL: This means block commitments will be incorrect"
    echo "Check test_inner_hash_count.log for details"
    exit 1
fi

echo ""
echo "=== Test 2: Full NoisyGEMM with Reduction ==="
echo "Testing complete mining path: noise + GEMM + denoise + inner hash"
python3 -m pytest tests/test_pearl_gemm.py -k "noisy_gemm" -v --tb=short 2>&1 | tee ../../test_full_noisy_gemm.log
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✓ Full NoisyGEMM PASSED"
else
    echo "✗ NoisyGEMM FAILED"
    echo "Check test_full_noisy_gemm.log"
    exit 1
fi

cd ../..

echo ""
echo "=== Mining Proof Path Validated ==="
echo ""
echo "What was tested:"
echo "  ✓ Inner hash / transcript extraction (bit-exact)"
echo "  ✓ NoisyGEMM with reduction enabled"
echo "  ✓ Full mining kernel path"
echo ""
echo "What still needs validation:"
echo "  - Commitment hash computation (Merkle + V3 salting)"
echo "  - ZK proof generation"
echo "  - Actual share acceptance on simnet"
echo ""
echo "Next: Run test_mining_simnet.sh for end-to-end mining test"
