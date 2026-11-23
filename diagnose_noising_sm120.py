#!/usr/bin/env python3
"""Diagnostic for sm_120 noising - bit-exact integer validation."""
import sys, torch

def test_noising():
    if not torch.cuda.is_available():
        print("✗ SKIP: No CUDA"); return 2
    
    cc = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name()} sm_{cc[0]}{cc[1]}")
    
    from pearl_gemm import noise_A, noise_B
    from pearl_gemm.testing import GEMMParam, GemmTensorGenerator
    from pearl_gemm_build_utils.kernel_configs.default_compiled_kernels import KERNEL_CONFIGS
    
    configs = [c for c in KERNEL_CONFIGS.matmul_kernels if c.pipeline_stages == 2]
    if not configs: print("✗ No stages=2 configs"); return 1
    mc = min(configs, key=lambda c: c.tile_size_m * c.tile_size_n)
    
    gp = GEMMParam(m=64, n=64, k=64, R=mc.R, matmul_config=mc)
    tg = GemmTensorGenerator(gp); tg.generate()
    
    print("Test 1: Noising A")
    EAL_EAR_cpu = torch._int_mm(tg.EAL.cpu().int8(), tg.EAR_K_major.cpu().int8().t())
    ApEA_ref = (tg.A.cpu().int() + EAL_EAR_cpu).to(torch.int8)
    tg.ApEA.zero_()
    noise_A(A=tg.A, EAL=tg.EAL, EAR_K_major=tg.EAR_K_major, ApEA=tg.ApEA,
            AxEBL_int32=tg.AxEBL_int32, EBL_R_major=tg.EBL_R_major,
            tile_size_m=mc.tile_size_m, tile_size_k=mc.tile_size_k)
    torch.cuda.synchronize()
    diff_a = (tg.ApEA.cpu().int() - ApEA_ref.int()).abs().max().item()
    if diff_a != 0: print(f"✗ FAIL: diff={diff_a}"); return 1
    print("✓ Noising A bit-exact")
    
    print("Test 2: Noising B")
    EBR_EBL_cpu = torch._int_mm(tg.EBR.cpu().int8(), tg.EBL_R_major.cpu().int8().t())
    BpEB_ref = (tg.B.cpu().int() + EBR_EBL_cpu).to(torch.int8)
    tg.BpEB.zero_()
    noise_B(B=tg.B, EBR=tg.EBR, EBL_R_major=tg.EBL_R_major, BpEB=tg.BpEB,
            EARxBpEB_int32=tg.EARxBpEB_int32, EAR_K_major=tg.EAR_K_major,
            tile_size_n=mc.tile_size_n, tile_size_k=mc.tile_size_k)
    torch.cuda.synchronize()
    diff_b = (tg.BpEB.cpu().int() - BpEB_ref.int()).abs().max().item()
    if diff_b != 0: print(f"✗ FAIL: diff={diff_b}"); return 1
    print("✓ Noising B bit-exact")
    
    print("\n✓ ALL PASSED (integer-exact)"); return 0

if __name__ == "__main__":
    try: sys.exit(test_noising())
    except Exception as e: print(f"✗ FATAL: {e}"); sys.exit(1)
