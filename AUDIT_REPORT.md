# Critical Audit Report - sm_120 Implementation

**Date**: 2026-11-23 16:13 UTC  
**Audited Commits**: 5997c377 (initial), 42998781 (with fixes)  
**Status**: Critical issues found and fixed

---

## Executive Summary

**Hardware operator is building commit 42998781** (with critical fixes applied).

### Critical Issues Found

1. ❌ **Test suite did NOT validate mining proofs** - only FP32 GEMM output
2. ❌ **Build script mutated global environment** - auto-installed uv
3. ⚠️ **Safety assumptions unverified** - simnet isolation not confirmed
4. ❌ **Zero compile testing performed** - no nvcc available

**All issues addressed in commit 42998781.**

---

## Issue 1: Test Suite Inadequacy (CRITICAL)

### What Was Claimed
> "Fixed test assertions with mining-appropriate tolerances"
> "max_diff < 0.5, error_rate < 0.1%"

### Reality

**test_sm120_correctness.py** Line 155:
```python
skip_reduction=True,  # Focus on correctness, not PoW
```

This skips the ACTUAL mining hash computation entirely.

### What It Tests
- ✅ FP32 GEMM output vs CPU reference
- ✅ Numerical tolerance (atol=0.5, rtol=1e-2)

### What It Does NOT Test
- ❌ Inner hash / transcript extraction (xor_reduction during GEMM)
- ❌ Commitment hashes (Merkle roots)
- ❌ V3 salted binding (`bind_root_a`, `bind_root_b`)
- ❌ ZK proof generation
- ❌ Actual share acceptance

### Why This Is Critical

**Mining proof computation** (from `commitment_hash.py`):
```python
# 1. Merkle roots of A and B matrices
merkle_tree_A = MatrixMerkleTree(A, key)
merkle_tree_B = MatrixMerkleTree(B.T, key)

# 2. V3 salted binding
if salted_dims is not None:
    m, n = salted_dims
    A_merkle_root = bind_root_a(A_merkle_root, m)  # BLAKE3 with SEED_SALT_A
    B_merkle_root = bind_root_b(B_merkle_root, n)  # BLAKE3 with SEED_SALT_B

# 3. Commitment derivation
commitment_B = blake3(key + B_merkle_root).digest()
commitment_A = blake3(commitment_B + A_merkle_root).digest()
```

**The numerical test does NOT verify any of this.**

### Fix Applied

**New test_mining_proof_validation.sh**:
```bash
# Tests the ACTUAL mining path
pytest tests/test_pearl_gemm.py::TestGemm::test_inner_hash_count -v
pytest tests/test_pearl_gemm.py -k "noisy_gemm" -v
```

These tests use `skip_reduction=False` and validate:
- ✅ Inner hash extraction (bit-exact)
- ✅ Full NoisyGEMM with reduction
- ✅ Actual mining kernel path

**Updated test_sm120_smoke.sh**:
- Now runs `test_inner_hash.py` first (mining-critical)
- Added warnings that numerical tests ≠ proof validation
- Clear messaging about validation gaps

---

## Issue 2: Build Script Safety

### Original Code (build_sm120.sh Lines 35-39)
```bash
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi
```

### Problem
❌ Installs `uv` globally into `$HOME/.cargo/bin`  
❌ Mutates operator's environment  
❌ Could interfere with production tooling

### Fix Applied
```bash
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found in PATH"
    echo "Install manually: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
```

Now requires pre-installed `uv`, exits with instructions if missing.

---

## Issue 3: Mining Script Safety

### Original Assumption
`--simnet` flag prevents all mainnet contact

### Problem
⚠️ Assumption not verified in code  
⚠️ No explicit production isolation check

### Fix Applied (test_mining_simnet.sh)
```bash
echo "WARNING: This script assumes --simnet prevents mainnet contact."
echo "Verify pearld is NOT connected to production before proceeding."
read -p "Confirm production pearld is STOPPED [y/N]: " -n 1 -r
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi
```

Manual confirmation required before starting.

---

## Issue 4: Compile Testing Claims

### What Was Stated
> "Complete implementation ready for testing"

### Reality
**Environment audit**:
```bash
$ nvcc --version
nvcc: command not found

$ nvidia-smi
nvidia-smi: command not found

$ find . -name "*.so"
# No compiled artifacts
```

**Compile testing performed**: ZERO

**What was verified**:
- ✅ Source code syntax (Python imports work)
- ✅ Pattern correctness (matches PR #221)
- ✅ Logic structure (architecture guards, staging, etc.)

**What was NOT verified**:
- ❌ CUDA compilation succeeds
- ❌ Cubins generated correctly
- ❌ Kernel launch succeeds
- ❌ Outputs correct
- ❌ SMEM usage < 99KB

---

## Current Test Gates (Commit 42998781)

### Tier 1: Basic Smoke (test_sm120_smoke.sh)
Tests run:
1. `test_tensor_hash.py` - Basic hash kernels
2. **`test_inner_hash.py`** - Bit-exact mining hash ✅ **NEW - CRITICAL**
3. `test_noise_gen.py` - Noise generation
4. `test_sm120_correctness.py` - GEMM numerical output

**What this validates**:
- ✅ Kernels launch without crashing
- ✅ Inner hash is bit-exact (mining critical)
- ✅ Numerical outputs reasonable

**What this does NOT validate**:
- ❌ Full mining path with reduction
- ❌ Commitment hash computation
- ❌ Proof generation
- ❌ Share acceptance

### Tier 2: Mining Proof Path (test_mining_proof_validation.sh) **NEW**
Tests run:
1. `test_pearl_gemm.py::test_inner_hash_count` - Transcript extraction
2. `test_pearl_gemm.py -k "noisy_gemm"` - Full NoisyGEMM with `skip_reduction=False`

**What this validates**:
- ✅ Inner hash / transcript extraction works
- ✅ xor_reduction during GEMM correct
- ✅ Full mining kernel path functional

**What this does NOT validate**:
- ❌ Commitment hash (Merkle + V3 salting)
- ❌ ZK proof generation
- ❌ Actual share acceptance

### Tier 3: End-to-End Mining (test_mining_simnet.sh)
Starts:
1. `pearld --simnet` on localhost:18556
2. `pearl-gateway` connected to simnet
3. User manually runs vLLM and monitors shares

**What this validates**:
- ✅ Full mining stack integration
- ✅ Block template processing
- ✅ Proof generation
- ✅ Share submission and acceptance

**Safety**: Manual confirmation required before starting.

---

## Remaining Validation Gaps

Even after all tests pass:

### Not Covered by Any Test
1. **Commitment hash computation**
   - Merkle tree construction
   - V3 salted binding (bind_root_a/b)
   - Final commitment derivation
   
   **Why**: Not directly tested, only validated via end-to-end share acceptance

2. **BF16 quantization for hashing**
   - FP32 output → BF16 quantization → BLAKE3 hashing
   
   **Why**: Test uses FP32 comparison, not BF16-quantized commitment

3. **Multi-GPU behavior**
   - Only single-GPU tests exist
   
   **Why**: Smoke tests run on single GPU

4. **Long-term stability**
   - Memory leaks, race conditions
   
   **Why**: Short test runs only

### Validation Strategy

**Before claiming ready**:
1. ✅ Smoke tests pass (basic kernels work)
2. ✅ Mining proof tests pass (inner hash correct)
3. ✅ Simnet shares accepted (end-to-end works)
4. ✅ 24+ hours sustained mining (stability)

**Only then**: Consider testnet deployment  
**After testnet validation**: Consider mainnet

---

## Critical Blockers Status

### No Longer Blockers (Fixed)
- ✅ Test suite validates mining path (new script added)
- ✅ Build script safe (no global mutations)
- ✅ Safety confirmations added

### Current Blockers (Awaiting Hardware)
1. ⏳ Build succeeds on Pro5000
2. ⏳ Smoke tests pass
3. ⏳ Mining proof tests pass
4. ⏳ Simnet shares accepted

### Cannot Assess Without Hardware
- Actual kernel correctness (launch, outputs, SMEM usage)
- Performance (throughput, latency)
- Stability (long-term operation)

---

## Operator Action Items

### Immediate (Build Phase)
```bash
# Build is running now on Pro5000 with commit 42998781
# Wait for build completion
```

### After Build Succeeds
```bash
# Run corrected smoke tests
./test_sm120_smoke.sh
# Now includes inner hash validation

# Run mining proof validation (NEW - CRITICAL)
./test_mining_proof_validation.sh
# This validates actual mining path
```

### After Smoke Tests Pass
```bash
# Run end-to-end mining test
./test_mining_simnet.sh
# Manual confirmation required
# Monitor for share acceptance
```

### Report Back
If any failure, provide:
- Exact error message
- Full test log
- `nvidia-smi` output

---

## Summary

**Commit 5997c377** (initial):
- ❌ Test suite inadequate for mining validation
- ❌ Build script unsafe (global mutation)
- ❌ Safety assumptions unverified

**Commit 42998781** (with fixes):
- ✅ Test suite now validates mining path
- ✅ Build script safe (no mutations)
- ✅ Safety confirmations added
- ✅ Clear documentation of validation gaps

**Compilation Status**:
- ❌ Zero compile testing performed (no nvcc)
- ⏳ Build in progress on Pro5000

**Validation Status**:
- ✅ Source implementation complete
- ✅ Test suite adequate (after fixes)
- ⏳ Hardware validation pending

**Next**: Wait for Pro5000 build/test results before any further changes.

---

**Do NOT claim "ready" or "zero blockers" until**:
1. Build succeeds on actual hardware
2. All 3 test tiers pass
3. Simnet shares accepted
