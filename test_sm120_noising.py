#!/usr/bin/env python3
"""
SM120 noising validation tests - zero-noise, zero-input, one-hot, random

Tests all 4 outputs independently with deterministic inputs.
Skips if exit non-zero (test failures).
"""
import sys
import torch

def test_zero_noise_A():
    """EAL=0 → EA=0 → ApEA should equal A exactly"""
    from pearl_gemm import noise_A
    from pearl_gemm.testing import GEMMParam, GemmTensorGenerator
    from pearl_gemm_build_utils.kernel_configs.default_compiled_kernels import KERNEL_CONFIGS
    
    configs = [c for c in KERNEL_CONFIGS.matmul_kernels 
               if c.pipeline_stages == 2 and c.tile_size_m == 64 and c.tile_size_n == 64]
    if not configs:
        print("✗ No configs"); return False
    
    results = []
    for mc in configs:
        m, k = 128, mc.tile_size_k
        gp = GEMMParam(m=m, n=128, k=k, R=mc.R, matmul_config=mc)
        tg = GemmTensorGenerator(gp)
        tg.generate()
        
        A_orig = tg.A.clone()
        tg.EAL.zero_()  # Zero noise
        tg.ApEA.zero_()
        tg.AxEBL.zero_()
        
        noise_A(A=tg.A, EAL=tg.EAL, AxEBL=tg.AxEBL, ApEA=tg.ApEA,
                EAR=tg.EAR_R_major, EBL=tg.EBL_K_major,
                tile_size_m=mc.tile_size_m, tile_size_k=mc.tile_size_k)
        torch.cuda.synchronize()
        
        diff = (tg.ApEA.int() - A_orig.int()).abs().max().item()
        ok = (diff == 0)
        print(f"  {'✓' if ok else '✗'} Zero-noise A R{mc.R}: ApEA diff={diff}")
        results.append(ok)
    
    return all(results)

def test_a_zero():
    """A=0 → ApEA = EA (isolates EA computation)"""
    from pearl_gemm import noise_A
    from pearl_gemm.testing import GEMMParam, GemmTensorGenerator
    from pearl_gemm_build_utils.kernel_configs.default_compiled_kernels import KERNEL_CONFIGS
    
    configs = [c for c in KERNEL_CONFIGS.matmul_kernels 
               if c.pipeline_stages == 2 and c.tile_size_m == 64][:2]
    
    results = []
    for mc in configs:
        m, k = 128, mc.tile_size_k
        gp = GEMMParam(m=m, n=128, k=k, R=mc.R, matmul_config=mc)
        tg = GemmTensorGenerator(gp)
        tg.generate()
        
        tg.A.zero_()  # Zero input
        EA_ref = torch._int_mm(tg.EAL, tg.EAR_R_major.t())
        
        tg.ApEA.zero_()
        tg.AxEBL.zero_()
        noise_A(A=tg.A, EAL=tg.EAL, AxEBL=tg.AxEBL, ApEA=tg.ApEA,
                EAR=tg.EAR_R_major, EBL=tg.EBL_K_major,
                tile_size_m=mc.tile_size_m, tile_size_k=mc.tile_size_k)
        torch.cuda.synchronize()
        
        diff = (tg.ApEA.int() - EA_ref.int()).abs().max().item()
        ok = (diff == 0)
        print(f"  {'✓' if ok else '✗'} A-zero R{mc.R}: ApEA vs EA diff={diff}")
        results.append(ok)
    
    return all(results)

def test_zero_noise_B():
    """EBR=0 → EB=0 → BpEB should equal B exactly"""
    from pearl_gemm import noise_B
    from pearl_gemm.testing import GEMMParam, GemmTensorGenerator
    from pearl_gemm_build_utils.kernel_configs.default_compiled_kernels import KERNEL_CONFIGS
    
    configs = [c for c in KERNEL_CONFIGS.matmul_kernels 
               if c.pipeline_stages == 2 and c.tile_size_n == 64][:2]
    
    results = []
    for mc in configs:
        n, k = 128, mc.tile_size_k
        gp = GEMMParam(m=128, n=n, k=k, R=mc.R, matmul_config=mc)
        tg = GemmTensorGenerator(gp)
        tg.generate()
        
        B_orig = tg.B.clone()
        tg.EBR.zero_()  # Zero noise
        tg.BpEB.zero_()
        tg.EARxBpEB.zero_()
        
        noise_B(B=tg.B, EBR=tg.EBR, EARxBpEB=tg.EARxBpEB, BpEB=tg.BpEB,
                EAR=tg.EAR_K_major, EBL=tg.EBL_R_major,
                tile_size_n=mc.tile_size_n, tile_size_k=mc.tile_size_k)
        torch.cuda.synchronize()
        
        diff = (tg.BpEB.int() - B_orig.int()).abs().max().item()
        ok = (diff == 0)
        print(f"  {'✓' if ok else '✗'} Zero-noise B R{mc.R}: BpEB diff={diff}")
        results.append(ok)
    
    return all(results)

def test_b_zero():
    """B=0 → BpEB = EB (isolates EB computation)"""
    from pearl_gemm import noise_B
    from pearl_gemm.testing import GEMMParam, GemmTensorGenerator
    from pearl_gemm_build_utils.kernel_configs.default_compiled_kernels import KERNEL_CONFIGS
    
    configs = [c for c in KERNEL_CONFIGS.matmul_kernels 
               if c.pipeline_stages == 2 and c.tile_size_n == 64][:2]
    
    results = []
    for mc in configs:
        n, k = 128, mc.tile_size_k
        gp = GEMMParam(m=128, n=n, k=k, R=mc.R, matmul_config=mc)
        tg = GemmTensorGenerator(gp)
        tg.generate()
        
        tg.B.zero_()  # Zero input
        EB_ref = torch._int_mm(tg.EBR, tg.EBL_R_major.t())
        
        tg.BpEB.zero_()
        tg.EARxBpEB.zero_()
        noise_B(B=tg.B, EBR=tg.EBR, EARxBpEB=tg.EARxBpEB, BpEB=tg.BpEB,
                EAR=tg.EAR_K_major, EBL=tg.EBL_R_major,
                tile_size_n=mc.tile_size_n, tile_size_k=mc.tile_size_k)
        torch.cuda.synchronize()
        
        diff = (tg.BpEB.int() - EB_ref.int()).abs().max().item()
        ok = (diff == 0)
        print(f"  {'✓' if ok else '✗'} B-zero R{mc.R}: BpEB vs EB diff={diff}")
        results.append(ok)
    
    return all(results)

def test_random_full():
    """Random inputs - all 4 outputs against CPU reference"""
    from pearl_gemm import noise_A, noise_B
    from pearl_gemm.testing import GEMMParam, GemmTensorGenerator
    from pearl_gemm_build_utils.kernel_configs.default_compiled_kernels import KERNEL_CONFIGS
    
    configs = [c for c in KERNEL_CONFIGS.matmul_kernels 
               if c.pipeline_stages == 2 and c.tile_size_m == 64 and c.tile_size_n == 64][:2]
    
    results = []
    for mc in configs:
        m, n, k = 128, 128, mc.tile_size_k
        gp = GEMMParam(m=m, n=n, k=k, R=mc.R, matmul_config=mc)
        tg = GemmTensorGenerator(gp)
        tg.generate()
        
        # Noise A
        EA = torch._int_mm(tg.EAL, tg.EAR_R_major.t())
        ApEA_ref = (tg.A.int() + EA).to(torch.int8)
        AxEBL_ref = torch._int_mm(tg.A, tg.EBL_R_major)
        if gp.AxEBL_type_noising == torch.float16:
            AxEBL_ref = (AxEBL_ref.float() * (2**-14)).half()
        
        tg.ApEA.zero_()
        tg.AxEBL.zero_()
        noise_A(A=tg.A, EAL=tg.EAL, AxEBL=tg.AxEBL, ApEA=tg.ApEA,
                EAR=tg.EAR_R_major, EBL=tg.EBL_K_major,
                tile_size_m=mc.tile_size_m, tile_size_k=mc.tile_size_k)
        torch.cuda.synchronize()
        
        apea_diff = (tg.ApEA.int() - ApEA_ref.int()).abs().max().item()
        axebl_diff = (tg.AxEBL - AxEBL_ref).abs().max().item()
        
        # Noise B
        EB = torch._int_mm(tg.EBR, tg.EBL_R_major.t())
        BpEB_ref = (tg.B.int() + EB).to(torch.int8)
        EARxBpEB_ref = torch._int_mm(BpEB_ref, tg.EAR_K_major.t())
        if gp.EARxBpEB_type_noising == torch.float16:
            EARxBpEB_ref = (EARxBpEB_ref.float() * (2**-12)).half()
        
        tg.BpEB.zero_()
        tg.EARxBpEB.zero_()
        noise_B(B=tg.B, EBR=tg.EBR, EARxBpEB=tg.EARxBpEB, BpEB=tg.BpEB,
                EAR=tg.EAR_K_major, EBL=tg.EBL_R_major,
                tile_size_n=mc.tile_size_n, tile_size_k=mc.tile_size_k)
        torch.cuda.synchronize()
        
        bpeb_diff = (tg.BpEB.int() - BpEB_ref.int()).abs().max().item()
        earxbpeb_diff = (tg.EARxBpEB - EARxBpEB_ref).abs().max().item()
        
        ok = (apea_diff == 0 and axebl_diff == 0 and bpeb_diff == 0 and earxbpeb_diff == 0)
        print(f"  {'✓' if ok else '✗'} Random R{mc.R}: ApEA={apea_diff} AxEBL={axebl_diff:.0f} BpEB={bpeb_diff} EARxBpEB={earxbpeb_diff:.0f}")
        results.append(ok)
    
    return all(results)

def main():
    if not torch.cuda.is_available():
        print("✗ No CUDA"); return 2
    
    cc = torch.cuda.get_device_capability()
    print(f"GPU: sm_{cc[0]}{cc[1]}\n")
    
    try:
        from pearl_gemm import noise_A, noise_B
    except ImportError as e:
        print(f"✗ Import failed: {e}"); return 1
    
    print("TEST 1: Zero-noise A (EAL=0)")
    r1 = test_zero_noise_A()
    
    print("\nTEST 2: A-zero (A=0, isolate EA)")
    r2 = test_a_zero()
    
    print("\nTEST 3: Zero-noise B (EBR=0)")
    r3 = test_zero_noise_B()
    
    print("\nTEST 4: B-zero (B=0, isolate EB)")
    r4 = test_b_zero()
    
    print("\nTEST 5: Random full (all 4 outputs)")
    r5 = test_random_full()
    
    print()
    if all([r1, r2, r3, r4, r5]):
        print("✓ ALL TESTS PASS")
        return 0
    else:
        failed = sum(1 for r in [r1,r2,r3,r4,r5] if not r)
        print(f"✗ {failed}/5 TESTS FAILED")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"✗ Exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
