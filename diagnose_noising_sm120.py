#!/usr/bin/env python3
"""sm_120 noising: authoritative layout, compiled configs only."""
import sys, torch

def test_one(mc, m, n, k, label):
    """Test (m,n,k) with config mc. Official formula per pearl_gemm.testing."""
    from pearl_gemm import noise_A, noise_B
    from pearl_gemm.testing import GEMMParam, GemmTensorGenerator
    
    gp = GEMMParam(m=m, n=n, k=k, R=mc.R, matmul_config=mc)
    tg = GemmTensorGenerator(gp); tg.generate()
    
    # Official A ref (test_pearl_gemm.py:195-212)
    # noise_A args: EAR=EAR_R_major (k,R), EBL=EBL_K_major (R,k)
    # Ref: EA = EAL @ EAR_R_major.T = (m,R) @ (k,R)^T = (m,R) @ (R,k) = (m,k)
    EA = torch._int_mm(tg.EAL, tg.EAR_R_major.clone())  # torch._int_mm(A,B) = A @ B.T
    ApEA_ref = (tg.A.int() + EA).to(torch.int8)
    # AxEBL = A @ EBL_K_major.T = (m,k) @ (R,k)^T = (m,k) @ (k,R) = (m,R)
    AxEBL_ref_i32 = torch._int_mm(tg.A, tg.EBL_K_major.clone())
    if gp.AxEBL_type_noising == torch.float16:
        AxEBL_ref = (AxEBL_ref_i32.float() * (2**-14)).half()
    else:
        AxEBL_ref = AxEBL_ref_i32
    
    tg.ApEA.zero_(); tg.AxEBL.zero_()
    noise_A(A=tg.A, EAL=tg.EAL, AxEBL=tg.AxEBL, ApEA=tg.ApEA,
            EAR=tg.EAR_R_major, EBL=tg.EBL_K_major,
            tile_size_m=mc.tile_size_m, tile_size_k=mc.tile_size_k)
    torch.cuda.synchronize()
    
    d_apea = (tg.ApEA.int() - ApEA_ref.int()).abs().max().item()
    d_axebl = (tg.AxEBL - AxEBL_ref).abs().max().item()
    tol_axebl = 1e-3 if gp.AxEBL_type_noising == torch.float16 else 0
    
    # Official B ref (test_pearl_gemm.py:174-193)
    # noise_B args: EAR=EAR_K_major (R,k), EBL=EBL_R_major (k,R)
    # EB = EBR @ EBL_R_major.T = (n,R) @ (k,R)^T = (n,R) @ (R,k) = (n,k)
    EB = torch._int_mm(tg.EBR, tg.EBL_R_major.clone())
    BpEB_ref = (tg.B.int() + EB).to(torch.int8)
    # EARxBpEB = BpEB @ EAR_R_major = (n,k) @ (R,k)^T (but EAR_R_major is (k,R))
    # Actually from code line 182: torch._int_mm(BpEB, EAR_ref) where EAR_ref=EAR_R_major (k,R)
    # So: BpEB @ EAR_R_major.T = (n,k) @ (k,R)^T = (n,k) @ (R,k) WRONG
    # Let me check: torch._int_mm(A,B) = A @ B.T
    # So torch._int_mm(BpEB, EAR_R_major) = BpEB @ EAR_R_major.T = (n,k) @ (k,R) = (n,R) ✓
    EARxBpEB_ref_i32 = torch._int_mm(BpEB_ref, tg.EAR_R_major.clone())
    if gp.EARxBpEB_type_noising == torch.float16:
        EARxBpEB_ref = (EARxBpEB_ref_i32.float() * (2**-12)).half()
    else:
        EARxBpEB_ref = EARxBpEB_ref_i32
    
    tg.BpEB.zero_(); tg.EARxBpEB.zero_()
    noise_B(B=tg.B, EBR=tg.EBR, EARxBpEB=tg.EARxBpEB, BpEB=tg.BpEB,
            EAR=tg.EAR_K_major, EBL=tg.EBL_R_major,
            tile_size_n=mc.tile_size_n, tile_size_k=mc.tile_size_k)
    torch.cuda.synchronize()
    
    d_bpeb = (tg.BpEB.int() - BpEB_ref.int()).abs().max().item()
    d_earxbpeb = (tg.EARxBpEB - EARxBpEB_ref).abs().max().item()
    tol_earxbpeb = 1e-3 if gp.EARxBpEB_type_noising == torch.float16 else 0
    
    ok = (d_apea == 0 and d_axebl <= tol_axebl and
          d_bpeb == 0 and d_earxbpeb <= tol_earxbpeb)
    status = "✓" if ok else "✗"
    print(f"  {status} {label:16s} ApEA={d_apea:3d} AxEBL={d_axebl:.2e} "
          f"BpEB={d_bpeb:3d} EARxBpEB={d_earxbpeb:.2e}")
    return ok

def main():
    if not torch.cuda.is_available(): print("✗ SKIP: No CUDA"); return 2
    cc = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name()} sm_{cc[0]}{cc[1]}\n")
    
    try:
        from pearl_gemm import noise_A, noise_B
        from pearl_gemm.testing import GEMMParam, GemmTensorGenerator
        from pearl_gemm_build_utils.kernel_configs.default_compiled_kernels import KERNEL_CONFIGS
    except ImportError as e: print(f"✗ IMPORT: {e}"); return 1
    
    # Only test compiled noising configs (stages=2, bM=bN=64)
    configs = [c for c in KERNEL_CONFIGS.matmul_kernels 
               if c.pipeline_stages == 2 and c.tile_size_m == 64 and c.tile_size_n == 64]
    if not configs: print("✗ No 64x64 stages=2 noising configs"); return 1
    
    results = []
    for mc in configs:
        R, bM, bN, bK = mc.R, mc.tile_size_m, mc.tile_size_n, mc.tile_size_k
        dtype = "fp16" if mc.convert_denoise else "int32"
        print(f"Config: {bM}×{bN}×{bK} R={R} {dtype}")
        
        # Tile-aligned sizes only (divisible by tile dims)
        sizes = [
            (64, 64, 64, f"R{R} {dtype} 1x1tile"),
            (128, 128, 128, f"R{R} {dtype} 2x2tile"),
            (64, 128, 64, f"R{R} {dtype} 1Mx2N"),
            (128, 64, 128, f"R{R} {dtype} 2Mx1N"),
        ]
        for m, n, k, lbl in sizes:
            if m % bM == 0 and n % bN == 0 and k % bK == 0:
                results.append(test_one(mc, m, n, k, lbl))
        print()
    
    if all(results):
        print(f"✓ ALL {len(results)} PASSED (int8 exact, fp16 ≤1e-3)")
        return 0
    else:
        fails = sum(not r for r in results)
        print(f"✗ FAILED {fails}/{len(results)}")
        return 1

if __name__ == "__main__":
    try: sys.exit(main())
    except Exception as e:
        print(f"✗ EXCEPTION: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
