# Pearl sm_120 GPU Testing Protocol

Comprehensive validation protocol for Pearl INT7 NoisyGEMM on Blackwell RTX PRO 5000 / RTX 50-series.

## Prerequisites

- Completed build from SM120_BUILD_GUIDE.md
- RTX PRO 5000 or RTX 50-series GPU available
- pearl-gemm compiled with `PEARL_GEMM_ARCH=sm_120a`
- Pearl node (pearld) available for integration tests

## Test Tier Structure

### Tier 1: Hash Kernels (Should Work Now)
Simple kernels without WGMMA/TMA. Expected to pass immediately.

### Tier 2: GEMM Kernels (Requires Full Implementation)
Complex kernels with mma.sync. Requires complete collective_mainloop/epilogue changes.

### Tier 3: Mining Integration (Final Validation)
End-to-end mining on simnet/testnet with block templates and proof generation.

---

## TIER 1: Hash Kernel Tests

**Status**: Should pass immediately after build  
**Risk**: Low (simple kernels)  
**Time**: ~2 minutes

### Test 1.1: Tensor Hash

```bash
cd /workspace
uv run pytest miner/pearl-gemm/tests/test_tensor_hash.py -v

# Expected output:
# test_tensor_hash.py::test_tensor_hash_basic PASSED
# test_tensor_hash.py::test_tensor_hash_large PASSED
# ... (all tests PASSED)
```

**What this tests**: BLAKE3-based tensor hashing kernel. No matrix operations.

**If it fails**:
- Check CUDA_HOME is set correctly
- Verify GPU is detected: `nvidia-smi`
- Check driver version: `nvidia-smi | grep Driver`
- Try with CUDA_LAUNCH_BLOCKING=1 for detailed errors

### Test 1.2: Inner Hash

```bash
uv run pytest miner/pearl-gemm/tests/test_inner_hash.py -v

# Expected output:
# test_inner_hash.py::test_inner_hash_correctness PASSED
# test_inner_hash.py::test_inner_hash_determinism PASSED
```

**What this tests**: Inner hash accumulation for PoW extraction.

### Test 1.3: Noise Generation

```bash
uv run pytest miner/pearl-gemm/tests/test_noise_gen.py -v

# Expected output:
# test_noise_gen.py::test_noise_generation_basic PASSED
# test_noise_gen.py::test_noise_generation_shapes PASSED
```

**What this tests**: Random noise matrix generation (EAL, EAR, EBL, EBR).

### Tier 1 Success Criteria

✅ All hash/noise tests pass  
✅ No "unspecified launch failure" errors  
✅ Outputs match CPU reference

**If Tier 1 fails**: Basic CUDA setup issue, not architecture-specific. Debug CUDA environment before proceeding.

---

## TIER 2: GEMM Kernel Tests

**Status**: Will fail until collective changes complete  
**Risk**: Medium (complex kernels, proven in PR #221)  
**Time**: ~10-15 minutes

### Test 2.1: Clean GEMM (No Noise, No Denoise)

```bash
cd /workspace/miner/pearl-gemm

# Create clean GEMM test
cat > tests/test_clean_gemm_sm120.py <<'EOF'
"""Test clean (un-noised) GEMM for sm_120 debugging."""
import torch
from pearl_gemm import noisy_gemm
from pearl_gemm.testing import GEMMParam, GemmTensorGenerator
from pearl_gemm_build_utils.kernel_configs.default_compiled_kernels import KERNEL_CONFIGS

def test_clean_gemm():
    mc = KERNEL_CONFIGS.matmul_kernels[0]  # First available config
    gp = GEMMParam(
        m=256, n=256, k=256,
        skip_noising_a=True, skip_noising_b=True,
        matmul_config=mc,
        use_variable_scales=False,
        skip_reduction=True,
    )
    tg = GemmTensorGenerator(gp)
    tg.generate()
    
    # Seed noised operands with clean operands
    tg.ApEA.copy_(tg.A)
    tg.BpEB.copy_(tg.B)
    
    noisy_gemm(
        A=tg.A, B=tg.B,
        EAL=tg.EAL, EAL_fp16=tg.EAL_fp16,
        EAR_R_major=tg.EAR_R_major, EBL_R_major=tg.EBL_R_major,
        EAR_K_major=tg.EAR_K_major, EBL_K_major=tg.EBL_K_major,
        EBR=tg.EBR, EBR_fp16=tg.EBR_fp16,
        AxEBL_fp16=tg.AxEBL_fp16, EARxBpEB_fp16=tg.EARxBpEB_fp16,
        ApEA=tg.ApEA, BpEB=tg.BpEB,
        A_scales=tg.A_scales, B_scales=tg.B_scales,
        C=tg.C,
        host_signal_header_pinned=tg.host_signal_header_pinned,
        host_signal_sync=tg.host_signal_sync,
        AxEBL_int32=tg.AxEBL_int32, EARxBpEB_int32=tg.EARxBpEB_int32,
        tile_size_m=gp.tile_size_m, tile_size_n=gp.tile_size_n,
        tile_size_k=gp.tile_size_k, pipeline_stages=gp.pipeline_stages,
        cluster_size_m=gp.cluster_size_m, cluster_size_n=gp.cluster_size_n,
        swizzle=gp.swizzle, swizzle_n_maj=gp.swizzle_n_maj,
        tile_size_m_noising_A=gp.tile_size_m_noising_A,
        tile_size_n_noising_B=gp.tile_size_n_noising_B,
        tile_size_k_noising_A=gp.tile_size_k_noising_A,
        tile_size_k_noising_B=gp.tile_size_k_noising_B,
        k_blocks_per_split_noising_A=gp.k_blocks_per_split_noising_A,
        k_blocks_per_split_noising_B=gp.k_blocks_per_split_noising_B,
        run_noising_A=False, run_noising_B=False,
        skip_reduction=True, skip_denoising=True,
        pow_target=tg.pow_target, pow_key=tg.pow_key,
    )
    torch.cuda.synchronize()
    
    # Verify against CPU reference
    AB = torch._int_mm(tg.A.cpu(), tg.B.cpu().t())
    ref = torch.einsum("mn,m,n->mn", AB.float(), tg.A_scales.cpu(), tg.B_scales.cpu())
    
    out = tg.C.cpu().float()
    assert torch.allclose(out, ref, atol=0.1, rtol=1e-2), \
        f"Clean GEMM failed: max_diff={((out - ref).abs().max())}"
    print(f"✓ Clean GEMM PASSED for {mc.tile_size_m}x{mc.tile_size_n}x{mc.tile_size_k}")

if __name__ == "__main__":
    test_clean_gemm()
EOF

uv run python tests/test_clean_gemm_sm120.py
```

**Expected (before collective changes)**:
```
RuntimeError: CUDA error: unspecified launch failure
  at csrc/gemm/pearl_noisingB_host.h:53
```

**Expected (after collective changes)**:
```
✓ Clean GEMM PASSED for 128x128x64
```

### Test 2.2: Full NoisyGEMM Correctness

```bash
uv run python miner/pearl-gemm/tests/test_sm120_correctness.py

# Expected output (after collective changes):
# ======================================================================
# Pearl INT7 NoisyGEMM sm_120 Correctness Test Suite
# ======================================================================
# Device: NVIDIA RTX PRO 5000 Ada Generation (sm_120)
# PyTorch: 2.5.1+cu124
# CUDA: 12.4
#
# Testing 12 kernel configurations...
#
#  1. ✓ PASS 64x128x64 R64 stages=2                   max_diff=0.085
#  2. ✓ PASS 64x128x64 R128 stages=2                  max_diff=0.092
#  3. ✓ PASS 128x128x64 R64 stages=2                  max_diff=0.088
#  4. ✓ PASS 128x128x64 R128 stages=2                 max_diff=0.091
# ...
# ======================================================================
# RESULTS: 12/12 configurations passed
# ======================================================================
# ✓ All tests PASSED - sm_120 implementation is correct!
```

### Test 2.3: Full Pearl GEMM Suite

```bash
# Run comprehensive test suite (skip slow/integration tests)
uv run pytest miner/pearl-gemm/tests/test_pearl_gemm.py -v \
  -m "not slow and not integration" \
  --ignore=test_vllm_execution.py

# Expected: ~100+ tests PASSED
```

### Tier 2 Success Criteria

✅ Clean GEMM passes for all tile configurations  
✅ Full NoisyGEMM passes (noising + GEMM + denoising)  
✅ No correctness failures (max_diff < 0.1 for FP16)  
✅ No SMEM overflow errors

**If Tier 2 fails**:
1. **"unspecified launch failure"**: Collective changes incomplete, review collective_mainloop/epilogue.hpp
2. **Correctness failures**: Check operand staging, verify ldmatrix copies are correct
3. **SMEM overflow**: Verify tile configurations are using stages=2 and fit in 99KB

---

## TIER 3: Mining Integration Tests

**Status**: After Tier 2 passes  
**Risk**: Low (kernel-independent protocol)  
**Time**: ~30 minutes

### Test 3.1: Simnet Setup

```bash
# Terminal 1: Start pearld in simnet mode
cd /workspace/bin
./pearld \
  --simnet \
  --rpcuser=testuser \
  --rpcpass=testpass \
  --rpclisten=127.0.0.1:18556 \
  --miningaddr=<TAPROOT_ADDRESS> \
  --debuglevel=debug \
  --logdir=/tmp/pearld-simnet

# Wait for "Chain state fully synced" message
```

```bash
# Terminal 2: Start pearl-gateway
export PEARLD_RPC_URL="http://127.0.0.1:18556"
export PEARLD_RPC_USER="testuser"
export PEARLD_RPC_PASSWORD="testpass"
export PEARLD_MINING_ADDRESS="<TAPROOT_ADDRESS>"

pearl-gateway start

# Expected output:
# [INFO] Connected to pearld simnet
# [INFO] Mining address: <address>
# [INFO] Listening on /tmp/pearlgw.sock
```

### Test 3.2: vLLM Miner Smoke Test

```bash
# Terminal 3: Start vLLM miner with 8B model
cd /workspace

uv run vllm serve pearl-ai/Llama-3.3-8B-Instruct-pearl \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.7 \
  --enforce-eager \
  --trust-remote-code

# Wait for model load, then check logs for:
# - "PearlKernel initialized" on each layer
# - "mining_enabled=True"
# - No CUDA errors
```

### Test 3.3: Generate Test Block

```bash
# Terminal 4: Send test inference request
curl http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "pearl-ai/Llama-3.3-8B-Instruct-pearl",
    "prompt": "What is proof of useful work?",
    "max_tokens": 128,
    "temperature": 0.7
  }'

# Monitor pearl-gateway logs for:
# - Block template received
# - NoisyGEMM operations executed
# - ZK proof generation attempt
# - Block submission (or PoW search)
```

### Test 3.4: Verify Block Template Processing

```bash
# Check pearl-gateway status
curl http://127.0.0.1:8339/status

# Expected JSON:
# {
#   "connected": true,
#   "mining_address": "<address>",
#   "blocks_processed": 1,
#   "last_template_time": "2026-11-23T...",
#   "proof_generation": "enabled"
# }
```

### Test 3.5: Testnet Mining (Optional)

**⚠️ TESTNET ONLY - DO NOT USE ON MAINNET UNTIL FULLY VALIDATED**

```bash
# Terminal 1: Connect to Pearl testnet
./bin/pearld \
  --testnet \
  --rpcuser=testuser \
  --rpcpass=testpass \
  --rpclisten=127.0.0.1:44109 \
  --miningaddr=<TESTNET_ADDRESS> \
  --addpeer=testnet-seed.pearl.network:44110

# Terminal 2: pearl-gateway (testnet config)
export PEARLD_RPC_URL="http://127.0.0.1:44109"
pearl-gateway start

# Terminal 3: vLLM miner
uv run vllm serve pearl-ai/Llama-3.3-70B-Instruct-pearl \
  --port 8000 --max-model-len 4096 \
  --tensor-parallel-size 2  # For dual-GPU if available
```

**Monitor for**:
- Block templates with real difficulty
- Valid share submissions
- Testnet block acceptance
- No protocol errors

### Tier 3 Success Criteria

✅ Simnet: Blocks generated and accepted locally  
✅ ZK proofs generated without errors  
✅ NoisyGEMM executes during inference  
✅ Gateway communicates with node successfully  
✅ (Testnet): Shares accepted by network

**If Tier 3 fails**:
- **No block templates**: Check pearld RPC connection, mining address
- **Proof generation fails**: Check ZK circuit compatibility (not sm_120-specific)
- **Share rejected**: Verify protocol version matches network
- **NoisyGEMM not running**: Check vLLM plugin loaded correctly

---

## Detailed Logging & Debugging

### Enable Verbose CUDA Logs

```bash
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_DEBUG=INFO  # If using tensor parallelism

# Re-run failing test
uv run python miner/pearl-gemm/tests/test_sm120_correctness.py
```

### Capture Kernel Launch Traces

```bash
# Profile a single test case
uv run python -c "
import torch
import pearl_gemm
from pearl_gemm.testing import GEMMParam, GemmTensorGenerator
from pearl_gemm_build_utils.kernel_configs.default_compiled_kernels import KERNEL_CONFIGS

torch.cuda.profiler.start()
mc = KERNEL_CONFIGS.matmul_kernels[0]
gp = GEMMParam(256, 256, 256, matmul_config=mc, skip_reduction=True)
tg = GemmTensorGenerator(gp)
tg.generate()
# ... run noisy_gemm ...
torch.cuda.profiler.stop()
"
```

### Check SMEM Usage

```bash
# Verify SMEM fits in 99KB
uv run python -c "
from pearl_gemm.csrc.gemm.kernel_traits import KernelTraits
# Calculate SharedStorage size for each config
# Should be < 99 * 1024 bytes
"

# Or check at runtime with CUDA occupancy calculator
```

### Collect Diagnostic Bundle

```bash
# Package logs and environment for support
cd /workspace
mkdir -p pearl-sm120-diagnostics

# System info
nvidia-smi > pearl-sm120-diagnostics/nvidia-smi.txt
nvcc --version > pearl-sm120-diagnostics/nvcc-version.txt
uv run python -c "
import torch, sys
with open('pearl-sm120-diagnostics/python-env.txt', 'w') as f:
    f.write(f'Python: {sys.version}\n')
    f.write(f'PyTorch: {torch.__version__}\n')
    f.write(f'CUDA: {torch.version.cuda}\n')
    f.write(f'Device: {torch.cuda.get_device_name(0)}\n')
    f.write(f'Compute: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}\n')
"

# Test results
uv run pytest miner/pearl-gemm/tests/ -v \
  --tb=short \
  > pearl-sm120-diagnostics/test-results.txt 2>&1

# Build artifacts
cp miner/pearl-gemm/setup.py pearl-sm120-diagnostics/
cuobjdump --list-elf $(find . -name "pearl_gemm_cuda*.so" | head -1) \
  > pearl-sm120-diagnostics/cuobjdump.txt

tar czf pearl-sm120-diagnostics.tar.gz pearl-sm120-diagnostics/
```

---

## Performance Benchmarking (Optional)

After correctness is validated, benchmark against Hopper:

```bash
# Throughput test
uv run pytest miner/pearl-gemm/tests/ -m performance -v

# vLLM end-to-end
uv run python -m vllm.benchmarks.benchmark_latency \
  --model pearl-ai/Llama-3.3-8B-Instruct-pearl \
  --input-len 1024 --output-len 128

# Compare sm_90a vs sm_120a
# Expected: sm_120 ~60-70% of sm_90a throughput (SM80 mma.sync vs WGMMA)
```

---

## Sign-off Checklist

Before deploying to production or mainnet:

- [ ] Tier 1 (Hash): All tests pass
- [ ] Tier 2 (GEMM): All correctness tests pass
- [ ] Tier 3 (Mining): Simnet blocks generated successfully
- [ ] Testnet: Multiple blocks accepted without errors
- [ ] No "unspecified launch failure" in any test
- [ ] SMEM usage verified < 99KB for all configs
- [ ] ZK proofs generate correctly
- [ ] Performance acceptable (compare to Hopper baseline)
- [ ] Rollback procedure documented and tested
- [ ] Diagnostic bundle collected
- [ ] All logs reviewed for warnings/errors

**Production deployment**: Only after ALL checklist items complete and sustained testnet operation (24+ hours) with no errors.

---

## Expected Timeline

| Phase | Time | Gating Factor |
|-------|------|---------------|
| Tier 1 | 5 min | Build complete |
| Tier 2 | 1-2 hrs | Collective changes applied |
| Tier 3 Simnet | 30 min | Tier 2 passes |
| Tier 3 Testnet | 24+ hrs | Tier 3 Simnet stable |
| Production | TBD | Testnet validation complete |

---

## Support & Next Steps

**If tests fail**: Review SM120_IMPLEMENTATION_STATUS.md for remaining work items.

**If tests pass**: Proceed to extended testnet validation, then mainnet deployment planning.

**For questions**: Document exact error messages, attach diagnostic bundle, reference this guide section.

This protocol ensures systematic validation from simple to complex, with clear success criteria at each tier.
