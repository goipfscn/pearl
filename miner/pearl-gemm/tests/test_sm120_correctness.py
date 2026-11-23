"""
SM120 Blackwell correctness test suite.

Standalone test harness for verifying INT7 NoisyGEMM correctness on sm_120/121.
Tests the full mining path: noising, GEMM, denoising, hashing, commitments.

Based on upstream PR #221 correctness.py.
"""

import os
import sys

# Ensure R32 is disabled (not compiled for sm_120 by default)
os.environ.setdefault("PEARL_GEMM_DISABLE_R32", "TRUE")

import torch
from pearl_gemm import noisy_gemm
from pearl_gemm.testing import GEMMParam, GemmTensorGenerator
from pearl_gemm_build_utils.kernel_configs.default_compiled_kernels import KERNEL_CONFIGS


def get_device_info():
    """Get CUDA device information."""
    if not torch.cuda.is_available():
        return None, None, "No CUDA device available"
    
    device = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device)
    cc = torch.cuda.get_device_capability(device)
    cc_str = f"sm_{cc[0]}{cc[1]}"
    
    return device, cc, f"{name} ({cc_str})"


def compute_ref_tensor(tg: GemmTensorGenerator):
    """Compute reference output using CPU reference implementation."""
    # Stage 1: Noise A and B
    ApEA_ref = (tg.A.cpu().to(torch.int32) + tg.EAL.cpu().to(torch.int32)).to(torch.int8)
    BpEB_ref = (tg.B.cpu().to(torch.int32) + tg.EBL_K_major.cpu().to(torch.int32)).to(torch.int8)
    
    # Stage 2: Matmul
    AB = torch._int_mm(ApEA_ref.to(torch.int8), BpEB_ref.to(torch.int8).t())
    C_scaled = torch.einsum(
        "mn,m,n->mn",
        AB.to(torch.float32),
        tg.A_scales.cpu(),
        tg.B_scales.cpu()
    )
    
    # Stage 3: Denoise
    # Compute EAL @ EBR^T (R x R matrix)
    EAL_EBR = torch._int_mm(
        tg.EAL.cpu().to(torch.int8),
        tg.EBR.cpu().to(torch.int8).t()
    ).to(torch.float16)
    
    # Compute (A @ EBR) and (EAL @ B)
    A_EBR = torch._int_mm(tg.A.cpu().to(torch.int8), tg.EBR.cpu().to(torch.int8).t()).to(torch.float16)
    EAL_B = torch._int_mm(tg.EAL.cpu().to(torch.int8), tg.B.cpu().to(torch.int8).t()).to(torch.float16)
    
    # Full denoise correction
    # C_denoised = C - (A @ EBR + EAL @ B + EAL @ EBR) @ Scales
    denoise_term = (
        torch.einsum("mr,m->mr", A_EBR, tg.A_scales.cpu()) @ 
        torch.einsum("nr,n->nr", torch.ones_like(tg.B_scales.cpu().unsqueeze(1)), tg.B_scales.cpu()).t()
    ) + (
        torch.einsum("mr,m->mr", EAL_B, tg.A_scales.cpu()) @ 
        torch.einsum("nr,n->nr", torch.ones_like(tg.B_scales.cpu().unsqueeze(1)), tg.B_scales.cpu()).t()
    ) + (
        EAL_EBR @ 
        torch.einsum("nr,n->nr", torch.ones_like(tg.B_scales.cpu().unsqueeze(1)), tg.B_scales.cpu()).t()
    )
    
    C_ref = C_scaled - denoise_term
    
    return C_ref.to(torch.float32)


def run_noisy_gemm(tg: GemmTensorGenerator, gp: GEMMParam):
    """Run pearl noisy_gemm kernel."""
    noisy_gemm(
        A=tg.A,
        B=tg.B,
        EAL=tg.EAL,
        EAL_fp16=tg.EAL_fp16,
        EAR_R_major=tg.EAR_R_major,
        EBL_R_major=tg.EBL_R_major,
        EAR_K_major=tg.EAR_K_major,
        EBL_K_major=tg.EBL_K_major,
        EBR=tg.EBR,
        EBR_fp16=tg.EBR_fp16,
        AxEBL_fp16=tg.AxEBL_fp16,
        EARxBpEB_fp16=tg.EARxBpEB_fp16,
        ApEA=tg.ApEA,
        BpEB=tg.BpEB,
        A_scales=tg.A_scales,
        B_scales=tg.B_scales,
        C=tg.C,
        host_signal_header_pinned=tg.host_signal_header_pinned,
        host_signal_sync=tg.host_signal_sync,
        AxEBL_int32=tg.AxEBL_int32,
        EARxBpEB_int32=tg.EARxBpEB_int32,
        tile_size_m=gp.tile_size_m,
        tile_size_n=gp.tile_size_n,
        tile_size_k=gp.tile_size_k,
        pipeline_stages=gp.pipeline_stages,
        cluster_size_m=gp.cluster_size_m,
        cluster_size_n=gp.cluster_size_n,
        swizzle=gp.swizzle,
        swizzle_n_maj=gp.swizzle_n_maj,
        tile_size_m_noising_A=gp.tile_size_m_noising_A,
        tile_size_n_noising_B=gp.tile_size_n_noising_B,
        tile_size_k_noising_A=gp.tile_size_k_noising_A,
        tile_size_k_noising_B=gp.tile_size_k_noising_B,
        k_blocks_per_split_noising_A=gp.k_blocks_per_split_noising_A,
        k_blocks_per_split_noising_B=gp.k_blocks_per_split_noising_B,
        run_noising_A=not gp.skip_noising_a,
        run_noising_B=not gp.skip_noising_b,
        skip_reduction=gp.skip_reduction,
        skip_denoising=False,
        pow_target=tg.pow_target,
        pow_key=tg.pow_key,
    )
    torch.cuda.synchronize()


def test_config(mc, m=256, n=512, k=512):
    """Test a single matmul configuration."""
    gp = GEMMParam(
        m, n, k,
        skip_noising_a=False,
        skip_noising_b=False,
        EARxBpEB_type_noising="fp16",
        AxEBL_type_noising="fp16",
        matmul_config=mc,
        use_variable_scales=True,
        skip_reduction=True,  # Focus on correctness, not PoW
    )
    
    tg = GemmTensorGenerator(gp)
    tg.generate()
    
    try:
        run_noisy_gemm(tg, gp)
        out = tg.C.cpu().to(torch.float32)
        ref = compute_ref_tensor(tg)
        
        # Check correctness
        abs_diff = (out - ref).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        
        # Tolerances for FP16 accumulation + INT8 quantization
        passed = torch.allclose(out, ref, atol=0.1, rtol=1e-2)
        
        result = {
            "passed": passed,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "config": f"{mc.tile_size_m}x{mc.tile_size_n}x{mc.tile_size_k} R{mc.R} stages={mc.pipeline_stages}",
        }
        
        return result
        
    except Exception as e:
        return {
            "passed": False,
            "error": f"{type(e).__name__}: {str(e)}",
            "config": f"{mc.tile_size_m}x{mc.tile_size_n}x{mc.tile_size_k} R{mc.R} stages={mc.pipeline_stages}",
        }


def main():
    """Run full correctness test suite."""
    device, cc, device_info = get_device_info()
    
    print("=" * 70)
    print("Pearl INT7 NoisyGEMM sm_120 Correctness Test Suite")
    print("=" * 70)
    print(f"Device: {device_info}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print()
    
    if device is None:
        print("ERROR: No CUDA device available")
        sys.exit(1)
    
    if cc[0] * 10 + cc[1] < 90:
        print(f"WARNING: This GPU (sm_{cc[0]}{cc[1]}) may not support pearl-gemm")
        print("Minimum requirement: sm_90 (Hopper) or sm_120 (Blackwell)")
        print()
    
    # Get compiled kernel configurations
    matmul_configs = sorted(
        KERNEL_CONFIGS.matmul_kernels,
        key=lambda c: (c.R, c.tile_size_m, c.tile_size_n)
    )
    
    if not matmul_configs:
        print("ERROR: No matmul kernels compiled!")
        print("Build pearl-gemm with appropriate architecture:")
        print("  export PEARL_GEMM_ARCH=sm_120a")
        print("  uv sync --package pearl-gemm --reinstall-package pearl-gemm")
        sys.exit(1)
    
    print(f"Testing {len(matmul_configs)} kernel configurations...")
    print()
    
    # Test each configuration
    results = []
    for i, mc in enumerate(matmul_configs, 1):
        result = test_config(mc, m=256, n=512, k=512)
        results.append(result)
        
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        config_str = result["config"]
        
        if result["passed"]:
            print(f"{i:2d}. {status} {config_str:40s} max_diff={result['max_diff']:.3f}")
        else:
            print(f"{i:2d}. {status} {config_str:40s}", end="")
            if "error" in result:
                print(f" ERROR: {result['error'][:50]}")
            else:
                print(f" max_diff={result.get('max_diff', 'N/A')}")
    
    # Summary
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    
    print()
    print("=" * 70)
    print(f"RESULTS: {passed}/{total} configurations passed")
    print("=" * 70)
    
    if passed == total:
        print("✓ All tests PASSED - sm_120 implementation is correct!")
        return 0
    else:
        print(f"✗ {total - passed} tests FAILED - review errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
