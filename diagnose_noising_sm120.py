#!/usr/bin/env python3
"""Diagnostic for sm_120 noising - isolates noising kernels only."""
import sys, subprocess, torch

def test_noising():
    if not torch.cuda.is_available(): return False
    from pearl_gemm import noise_A, noise_B
    from pearl_gemm.testing import GEMMParam, GemmTensorGenerator
    from pearl_gemm_build_utils.kernel_configs.default_compiled_kernels import KERNEL_CONFIGS
    
    configs = [c for c in KERNEL_CONFIGS.matmul_kernels if c.pipeline_stages == 2]
    mc = min(configs, key=lambda c: c.tile_size_m * c.tile_size_n)
    gp = GEMMParam(m=64, n=64, k=64, R=mc.R, matmul_config=mc)
    tg = GemmTensorGenerator(gp); tg.generate()
    
    print("Noising A...")
    noise_A(A=tg.A, EAL=tg.EAL, EAR_K_major=tg.EAR_K_major, ApEA=tg.ApEA,
            AxEBL_int32=tg.AxEBL_int32, EBL_R_major=tg.EBL_R_major,
            tile_size_m=mc.tile_size_m, tile_size_k=mc.tile_size_k)
    torch.cuda.synchronize()
    print("✓ Noising A OK")
    
    print("Noising B...")
    noise_B(B=tg.B, EBR=tg.EBR, EBL_R_major=tg.EBL_R_major, BpEB=tg.BpEB,
            EARxBpEB_int32=tg.EARxBpEB_int32, EAR_K_major=tg.EAR_K_major,
            tile_size_n=mc.tile_size_n, tile_size_k=mc.tile_size_k)
    torch.cuda.synchronize()
    print("✓ Noising B OK")
    return True

if __name__ == "__main__":
    try:
        test_noising()
        print("\n✓ All noising kernels passed")
    except RuntimeError as e:
        print(f"\n✗ FAILED: {e}")
        sys.exit(1)
