# Quick Start - RTX PRO 5000 Testing

**Commit**: `25f41533b6affcbe9161d0373d0680618d8b2687`  
**Branch**: `cursor/sm120-blackwell-support-1064`

## Immediate Actions

### 1. Clone and Build (15 minutes)
```bash
git clone https://github.com/goipfscn/pearl.git
cd pearl
git checkout cursor/sm120-blackwell-support-1064
git rev-parse HEAD  # Verify: 25f41533b6affcbe9161d0373d0680618d8b2687

./build_sm120.sh
# Builds for sm_120a, logs to build_sm120.log
# Exit code 0 = success
```

### 2. Smoke Test (1 minute)
```bash
./test_sm120_smoke.sh
# Runs: hash kernels, noise gen, GEMM correctness
# Exit code 0 = all tests passed
```

### 3. Mining Test (simnet)
```bash
export PEARLD_MINING_ADDRESS=$(./bin/prlctl getnewaddress)
./test_mining_simnet.sh
# Starts pearld + gateway in simnet mode
# Then start vLLM and monitor for shares
```

## Expected Results

### Build Success
```
=== Build Complete ===
Found: .../pearl_gemm_cuda.cpython-*.so
arch = sm_120
pearl_gemm: OK
pearl_gemm_cuda: OK
```

### Smoke Test Success
```
✓ Hash kernels PASSED
✓ Noise generation PASSED
✓ GEMM correctness PASSED
=== All Smoke Tests PASSED ===
```

## Failure Modes

### Build Fails
- Check `build_sm120.log`
- Verify CUDA 12.4+ installed: `nvcc --version`
- Check disk space (needs ~10GB during build)

### Smoke Test Fails
- **Hash/Noise fail**: Check GPU detected correctly
- **GEMM fails "unspecified launch failure"**: Kernel bug, report exact error
- **GEMM fails "max_diff > 0.5"**: Numerical issue, report max_diff value

### Mining Test Issues
- Check `pearld` and `gateway` both running: `ps aux | grep pearl`
- Check logs: `/tmp/pearld-simnet/`
- Verify mining address set

## Report Back

If build succeeds:
```bash
git rev-parse HEAD
ls -lh $(find . -name "pearl_gemm_cuda*.so")
```

If smoke test succeeds:
```bash
cat test_gemm.log | grep -A5 "results:"
```

If fails:
```bash
# Send exact error message and:
tail -50 build_sm120.log     # for build failure
tail -50 test_gemm.log       # for GEMM failure
nvidia-smi                   # GPU info
```

## Critical: No Mainnet

All tests are simnet/testnet ONLY. Do NOT point at production RPC.

## Fast Iteration

Branch stays same. Any fixes pushed to:
`https://github.com/goipfscn/pearl/tree/cursor/sm120-blackwell-support-1064`

Pull latest: `git pull origin cursor/sm120-blackwell-support-1064`
