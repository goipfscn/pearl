#!/usr/bin/env python3
"""sm_120 noising: R=64/128 sweep, rectangular M/N/K, all outputs exact."""
import sys, torch

def test_config(mc, m, n, k, label=""):
    """Test one (m,n,k) with given config."""
    from pearl_gemm import noise_A, noise_B
    from pearl_gemm.testing import GEMMParam, GemmTensorGenerator
    
    gp = GEMMParam(m=m, n=n, k=k, R=mc.R, matmul_config=mc)
    tg = GemmTensorGenerator(gp); tg.generate()
    
    # Ref A (per test_pearl_gemm.py:195-212)
    EA = torch._int_mm(tg.EAL, tg.EAR_K_major.t())
    ApEA_ref = (tg.A.int() + EA).to(torch.int8)
    AxEBL_ref_i32 = torch._int_mm(tg.A, tg.EBL_R_major)
    if gp.AxEBL_type_noising == torch.float16:
        AxEBL_ref = (AxEBL_ref_i32.float() * (2**-14)).half()
    else:
        AxEBL_ref = AxEBL_ref_i32
    
    tg.ApEA.zero_(); tg.AxEBL.zero_()
    noise_A(A=tg.A, EAL=tg.EAL, AxEBL=tg.AxEBL, ApEA=tg.ApEA,
            EAR=tg.EAR_K_major, EBL=tg.EBL_R_major,
            tile_size_m=mc.tile_size_m, tile_size_k=mc.tile_size_k)
    torch.cuda.synchronize()
    
    d_apea = (tg.ApEA.int() - ApEA_ref.int()).abs().max().item()
    d_axebl = (tg.AxEBL - AxEBL_ref).abs().max().item()
    tol_axebl = 1e-3 if gp.AxEBL_type_noising == torch.float16 else 0
    
    # Ref B (per test_pearl_gemm.py:174-193)
    EB = torch._int_mm(tg.EBR, tg.EBL_R_major.t())
    BpEB_ref = (tg.B.int() + EB).to(torch.int8)
    # EARxBpEB = BpEB @ EAR_R_major.T (n,k) @ (k,r) = (n,r)
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
    print(f"  {status} {label:12s} {m:3d}×{n:3d}×{k:3d}: "
          f"ApEA={d_apea} AxEBL={d_axebl:.1e} BpEB={d_bpeb} EARxBpEB={d_earxbpeb:.1e}")
    return ok

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
    
    results = []
    for mc in configs:
        R = mc.R
        print(f"\nConfig: {mc.tile_size_m}×{mc.tile_size_n}×{mc.tile_size_k} R={R}")
        # Square + rectangular (expose transpose/layout bugs)
        sizes = [
            (64, 64, 64, f"R{R} square"),
            (64, 128, 64, f"R{R} wide"),
            (128, 64, 64, f"R{R} tall"),
            (96, 80, 112, f"R{R} irregular"),
        ]
        for m, n, k, lbl in sizes:
            results.append(test_config(mc, m, n, k, lbl))
    
    if all(results):
        print(f"\n✓ ALL {len(results)} PASSED (int8 exact, fp16 ≤1e-3)")
        return 0
    else:
        fails = sum(not r for r in results)
        print(f"\n✗ FAILED {fails}/{len(results)}")
        return 1

if __name__ == "__main__":
    try: sys.exit(main())
    except Exception as e:
        print(f"✗ EXCEPTION: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
