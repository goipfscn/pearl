#!/usr/bin/env python3
"""
Test fragment layout compatibility with deterministic inputs.

Tests:
1. Zero-noise identity: EAL=0 or EAR=0 → EA=0 → ApEA should equal A exactly
2. A-zero product: A=0 → ApEA=EA, compare isolated EA
3. One-hot: Single non-zero element to track coordinate mapping
"""
import sys
import torch

def test_zero_noise_identity():
    """EAL=0 → EA=0 → ApEA=A (exact)"""
    from pearl_gemm import noise_A
    from pearl_gemm.testing import GEMMParam, GemmTensorGenerator
    from pearl_gemm_build_utils.kernel_configs.default_compiled_kernels import KERNEL_CONFIGS
    
    # Find first compiled 64x64 config
    configs = [c for c in KERNEL_CONFIGS.matmul_kernels 
               if c.pipeline_stages == 2 and c.tile_size_m == 64 and c.tile_size_n == 64]
    if not configs:
        print("✗ No configs"); return False
    
    mc = configs[0]
    m, n, k = 128, 128, mc.tile_size_k
    gp = GEMMParam(m=m, n=n, k=k, R=mc.R, matmul_config=mc)
    tg = GemmTensorGenerator(gp)
    
    # Generate base tensors
    tg.generate()
    A_orig = tg.A.clone()
    
    # TEST 1: EAL = 0 → EA = 0 → ApEA should equal A
    print(f"TEST 1: Zero-noise (EAL=0), m={m} k={k} R={mc.R}")
    tg.EAL.zero_()
    tg.ApEA.zero_()
    tg.AxEBL.zero_()
    
    noise_A(A=tg.A, EAL=tg.EAL, AxEBL=tg.AxEBL, ApEA=tg.ApEA,
            EAR=tg.EAR_R_major, EBL=tg.EBL_K_major,
            tile_size_m=mc.tile_size_m, tile_size_k=mc.tile_size_k)
    torch.cuda.synchronize()
    
    diff = (tg.ApEA.int() - A_orig.int()).abs()
    max_diff = diff.max().item()
    
    if max_diff == 0:
        print(f"  ✓ EXACT: ApEA == A when EAL=0")
        return True
    else:
        print(f"  ✗ FAIL: max_diff={max_diff}")
        # Show first mismatch
        idx = (diff > 0).nonzero()[0]
        i, j = idx[0].item(), idx[1].item()
        print(f"    First mismatch at ({i},{j}): A={A_orig[i,j]} ApEA={tg.ApEA[i,j]} diff={diff[i,j]}")
        return False

def test_a_zero():
    """A=0 → ApEA=EA, can isolate EA computation"""
    from pearl_gemm import noise_A
    from pearl_gemm.testing import GEMMParam, GemmTensorGenerator
    from pearl_gemm_build_utils.kernel_configs.default_compiled_kernels import KERNEL_CONFIGS
    
    configs = [c for c in KERNEL_CONFIGS.matmul_kernels 
               if c.pipeline_stages == 2 and c.tile_size_m == 64 and c.tile_size_n == 64]
    if not configs:
        return False
    
    mc = configs[0]
    m, n, k = 128, 128, mc.tile_size_k
    gp = GEMMParam(m=m, n=n, k=k, R=mc.R, matmul_config=mc)
    tg = GemmTensorGenerator(gp)
    tg.generate()
    
    # TEST 2: A = 0 → ApEA = EA
    print(f"TEST 2: A-zero (A=0), isolate EA")
    tg.A.zero_()
    tg.ApEA.zero_()
    tg.AxEBL.zero_()
    
    # Compute reference EA
    EA_ref = torch._int_mm(tg.EAL, tg.EAR_R_major.t())
    
    noise_A(A=tg.A, EAL=tg.EAL, AxEBL=tg.AxEBL, ApEA=tg.ApEA,
            EAR=tg.EAR_R_major, EBL=tg.EBL_K_major,
            tile_size_m=mc.tile_size_m, tile_size_k=mc.tile_size_k)
    torch.cuda.synchronize()
    
    # ApEA should equal EA
    diff = (tg.ApEA.int() - EA_ref.int()).abs()
    max_diff = diff.max().item()
    
    if max_diff == 0:
        print(f"  ✓ EXACT: ApEA == EA when A=0")
        return True
    else:
        print(f"  ✗ FAIL: max_diff={max_diff}")
        idx = (diff > 0).nonzero()[0]
        i, j = idx[0].item(), idx[1].item()
        print(f"    At ({i},{j}): EA_ref={EA_ref[i,j]} ApEA={tg.ApEA[i,j]}")
        return False

def main():
    if not torch.cuda.is_available():
        print("✗ No CUDA"); return 2
    
    cc = torch.cuda.get_device_capability()
    print(f"GPU: sm_{cc[0]}{cc[1]}\n")
    
    try:
        from pearl_gemm import noise_A
    except ImportError as e:
        print(f"✗ Import failed: {e}"); return 1
    
    results = []
    results.append(test_zero_noise_identity())
    results.append(test_a_zero())
    
    print()
    if all(results):
        print("✓ ALL EXACT")
        return 0
    else:
        print(f"✗ {sum(not r for r in results)}/{len(results)} FAILED")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"✗ {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
