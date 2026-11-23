#!/usr/bin/env python3
"""sm_120 noising diagnostic with validated reference."""
import sys, torch

def main():
    if not torch.cuda.is_available(): print("✗ SKIP: No CUDA"); return 2
    cc = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name()} sm_{cc[0]}{cc[1]}")
    
    try:
        from pearl_gemm import noise_A, noise_B
        from pearl_gemm.testing import GEMMParam, GemmTensorGenerator
        from pearl_gemm_build_utils.kernel_configs.default_compiled_kernels import KERNEL_CONFIGS
    except ImportError as e: print(f"✗ IMPORT: {e}"); return 1
    
    configs = [c for c in KERNEL_CONFIGS.matmul_kernels if c.pipeline_stages == 2]
    if not configs: print("✗ No stages=2"); return 1
    mc = min(configs, key=lambda c: c.tile_size_m * c.tile_size_n)
    print(f"Config: {mc.tile_size_m}x{mc.tile_size_n}x{mc.tile_size_k} R={mc.R}\n")
    
    gp = GEMMParam(m=64, n=64, k=64, R=mc.R, matmul_config=mc)
    tg = GemmTensorGenerator(gp); tg.generate()
    
    # Test A: ApEA = A + EAL×EAR, AxEBL = A×EBL
    print("Test: Noising A")
    EA = torch._int_mm(tg.EAL, tg.EAR_K_major.t())
    ApEA_ref = (tg.A.int() + EA).to(torch.int8)
    AxEBL_ref = torch._int_mm(tg.A, tg.EBL_R_major)
    if gp.AxEBL_type_noising == torch.float16:
        AxEBL_ref = (AxEBL_ref.float() * (2**-14)).half()
    
    tg.ApEA.zero_(); tg.AxEBL.zero_()
    noise_A(A=tg.A, EAL=tg.EAL, AxEBL=tg.AxEBL, ApEA=tg.ApEA,
            EAR=tg.EAR_K_major, EBL=tg.EBL_R_major,
            tile_size_m=mc.tile_size_m, tile_size_k=mc.tile_size_k)
    torch.cuda.synchronize()
    
    d_apea = (tg.ApEA.int() - ApEA_ref.int()).abs().max().item()
    d_axebl = (tg.AxEBL - AxEBL_ref).abs().max().item()
    tol_axebl = 0.01 if gp.AxEBL_type_noising == torch.float16 else 0
    if d_apea != 0 or d_axebl > tol_axebl:
        print(f"✗ FAIL: ApEA={d_apea} AxEBL={d_axebl}"); return 1
    print(f"✓ PASS: ApEA exact, AxEBL={d_axebl:.6f}")
    
    # Test B: BpEB = B + EBR×EBL
    print("Test: Noising B")
    EB = torch._int_mm(tg.EBR, tg.EBL_R_major.t())
    BpEB_ref = (tg.B.int() + EB).to(torch.int8)
    
    tg.BpEB.zero_(); tg.EARxBpEB.zero_()
    noise_B(B=tg.B, EBR=tg.EBR, EARxBpEB=tg.EARxBpEB, BpEB=tg.BpEB,
            EAR=tg.EAR_K_major, EBL=tg.EBL_R_major,
            tile_size_n=mc.tile_size_n, tile_size_k=mc.tile_size_k)
    torch.cuda.synchronize()
    
    d_bpeb = (tg.BpEB.int() - BpEB_ref.int()).abs().max().item()
    if d_bpeb != 0: print(f"✗ FAIL: BpEB={d_bpeb}"); return 1
    print(f"✓ PASS: BpEB exact")
    
    print("\n✓ ALL PASSED"); return 0

if __name__ == "__main__":
    try: sys.exit(main())
    except Exception as e: print(f"✗ EXCEPTION: {e}"); import traceback; traceback.print_exc(); sys.exit(1)
