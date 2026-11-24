#!/usr/bin/env python3
"""sm_120 noising: authoritative shapes, compiled configs, non-square test."""
import sys, torch

def test_one(mc, m, n, k, label):
    """Test (m,n,k) with config. Shapes verified per gemm_tensor_generator.py."""
    from pearl_gemm import noise_A, noise_B
    from pearl_gemm.testing import GEMMParam, GemmTensorGenerator
    
    gp = GEMMParam(m=m, n=n, k=k, R=mc.R, matmul_config=mc)
    tg = GemmTensorGenerator(gp); tg.generate()
    
    # Verified shapes:
    # EAR_R_major: (k, R), EBL_R_major: (k, R) [from gemm_tensor_generator.py:35]
    # torch._int_mm(A, B) = A @ B [regular matmul]
    
    # Noise A (test_pearl_gemm.py:195-212)
    # noise_A(EAR=EAR_R_major, EBL=EBL_K_major)
    EA = torch._int_mm(tg.EAL, tg.EAR_R_major.t())  # (m,R) @ (R,k) = (m,k)
    ApEA_ref = (tg.A.int() + EA).to(torch.int8)
    AxEBL = torch._int_mm(tg.A, tg.EBL_R_major)  # (m,k) @ (k,R) = (m,R)
    if gp.AxEBL_type_noising == torch.float16:
        AxEBL_ref = (AxEBL.float() * (2**-14)).half()
    else:
        AxEBL_ref = AxEBL
    
    tg.ApEA.zero_(); tg.AxEBL.zero_()
    noise_A(A=tg.A, EAL=tg.EAL, AxEBL=tg.AxEBL, ApEA=tg.ApEA,
            EAR=tg.EAR_R_major, EBL=tg.EBL_K_major,
            tile_size_m=mc.tile_size_m, tile_size_k=mc.tile_size_k)
    torch.cuda.synchronize()
    
    d_apea = (tg.ApEA.int() - ApEA_ref.int()).abs().max().item()
    d_axebl = (tg.AxEBL - AxEBL_ref).abs().max().item()
    
    # Noise B (test_pearl_gemm.py:174-193)
    # noise_B(EAR=EAR_K_major, EBL=EBL_R_major)
    EB = torch._int_mm(tg.EBR, tg.EBL_R_major.t())  # (n,R) @ (R,k) = (n,k)
    BpEB_ref = (tg.B.int() + EB).to(torch.int8)
    EARxBpEB = torch._int_mm(BpEB_ref, tg.EAR_R_major)  # (n,k) @ (k,R) = (n,R)
    if gp.EARxBpEB_type_noising == torch.float16:
        EARxBpEB_ref = (EARxBpEB.float() * (2**-12)).half()
    else:
        EARxBpEB_ref = EARxBpEB
    
    tg.BpEB.zero_(); tg.EARxBpEB.zero_()
    noise_B(B=tg.B, EBR=tg.EBR, EARxBpEB=tg.EARxBpEB, BpEB=tg.BpEB,
            EAR=tg.EAR_K_major, EBL=tg.EBL_R_major,
            tile_size_n=mc.tile_size_n, tile_size_k=mc.tile_size_k)
    torch.cuda.synchronize()
    
    d_bpeb = (tg.BpEB.int() - BpEB_ref.int()).abs().max().item()
    d_earxbpeb = (tg.EARxBpEB - EARxBpEB_ref).abs().max().item()
    
    ok = (d_apea == 0 and d_axebl == 0 and d_bpeb == 0 and d_earxbpeb == 0)
    status = "✓" if ok else "✗"
    print(f"  {status} {label:18s} ApEA={d_apea:3d} AxEBL={d_axebl:.0f} "
          f"BpEB={d_bpeb:3d} EARxBpEB={d_earxbpeb:.0f}")
    return ok

def main():
    if not torch.cuda.is_available(): print("✗ SKIP"); return 2
    cc = torch.cuda.get_device_capability()
    print(f"GPU: sm_{cc[0]}{cc[1]}\n")
    
    try:
        from pearl_gemm import noise_A, noise_B
        from pearl_gemm.testing import GEMMParam, GemmTensorGenerator
        from pearl_gemm_build_utils.kernel_configs.default_compiled_kernels import KERNEL_CONFIGS
    except ImportError as e: print(f"✗ IMPORT: {e}"); return 1
    
    # Compiled noising: 64×64 tiles, stages=2
    configs = [c for c in KERNEL_CONFIGS.matmul_kernels 
               if c.pipeline_stages == 2 and c.tile_size_m == 64 and c.tile_size_n == 64]
    if not configs: print("✗ No 64×64 configs"); return 1
    
    results = []
    for mc in configs:
        R, bK = mc.R, mc.tile_size_k
        dtype = "fp16" if mc.convert_denoise else "int32"
        print(f"Config: 64×64×{bK} R={R} {dtype}")
        
        # Tile-aligned + non-square (k≠R)
        for m, n, k, lbl in [
            (64, 64, 64, f"R{R} {dtype} square"),
            (128, 128, 128, f"R{R} {dtype} 2×2tile"),
            (64, 128, bK, f"R{R} {dtype} wide"),
            (128, 64, 2*bK, f"R{R} {dtype} tall"),
        ]:
            if m % 64 == 0 and n % 64 == 0 and k % bK == 0:
                results.append(test_one(mc, m, n, k, lbl))
        print()
    
    if all(results):
        print(f"✓ ALL {len(results)} EXACT"); return 0
    else:
        print(f"✗ FAILED {sum(not r for r in results)}/{len(results)}"); return 1

if __name__ == "__main__":
    try: sys.exit(main())
    except Exception as e: print(f"✗ {e}"); import traceback; traceback.print_exc(); sys.exit(1)
