# Compile Test Status - SHA 2137b2ff

## Changes Made

### 1. API Dispatch Fixed
- Removed stray `}` and `{` from broken branch structure
- Added `use_sm120_fallback` flag to noise_B
- Added `C10_CUDA_KERNEL_LAUNCH_CHECK()` after kernel launches
- Added `TORCH_CHECK` for required EAR/EBL in SM120 path
- Removed `cudaStreamSynchronize` (unnecessary, breaks graph capture)

### 2. Kernels Restored
- Reverted pearl_noisingA_kernel.h to cd691215 (last clean compile)
- Reverted pearl_noisingB_kernel.h to cd691215
- Removes e3582024 broken experiments:
  - Undefined `tiled_mma_mkr` (actual object: `tiled_mma`)
  - CuTe rank/index mismatches
  - False `make_tensor_like` alias assumptions
  - SMEM-add with dead storage writes

### 3. SM120 Launcher
- Removed unnecessary cudaStreamSynchronize between BpEB and EARxBpEB
- Kernels on same stream are automatically ordered
- Graph-capture compatible

## Compilation Status

**Local environment**: No torch installed (cloud VM)  
**Cannot test compile** without dependencies

**User must compile**:
```bash
cd /workspace/miner/pearl-gemm
export PEARL_GEMM_ARCH=sm_120a PEARL_GEMM_FORCE_BUILD=1 MAX_JOBS=4
rm -rf build *.so
python3 setup.py build_ext --inplace
```

## Code Review Verification

**API structure**: Clean (no stray braces)  
**Kernel includes**: Restored to working cd691215  
**SM120 launcher**: Syntax correct, semantics correct  
**Headers**: C10_CUDA_KERNEL_LAUNCH_CHECK available in ATen

## Next User Steps

1. Compile on hardware with torch/CUDA
2. Run test_sm120_noising.py
3. Verify all 5 tests pass
4. Run diagnose_noising_sm120.py for full sweep

Production remains PAUSED.
