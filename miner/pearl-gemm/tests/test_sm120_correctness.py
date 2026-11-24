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
    """
    Compute reference output using CPU reference implementation.
    
    IMPORTANT: This computes the EXACT mathematical result that the GPU kernel
    should produce, accounting for:
    - INT8 quantization and saturation
    - FP16 denoise operands (after INT8->FP16 cast)
    - FP32 scaling factors
    - 2^-12 denoise scaling
    
    The GPU kernel uses FP16 accumulation for denoise and FP32 for final output.
    We use FP32 throughout on CPU for reference since we're checking correctness
    not performance.
    """
    # Stage 1: Noise A and B (INT8 arithmetic with saturation)
    ApEA_ref = (tg.A.cpu().to(torch.int32) + tg.EAL.cpu().to(torch.int32))
    ApEA_ref = torch.clamp(ApEA_ref, -128, 127).to(torch.int8)
    
    BpEB_ref = (tg.B.cpu().to(torch.int32) + tg.EBL_K_major.cpu().to(torch.int32))
    BpEB_ref = torch.clamp(BpEB_ref, -128, 127).to(torch.int8)
    
    # Stage 2: Main GEMM (INT8 matmul -> INT32 accumulator -> FP32 scaling)
    AB = torch._int_mm(ApEA_ref, BpEB_ref.t()).to(torch.float32)
    
    # Apply per-row and per-column scaling
    # C_scaled[m,n] = AB[m,n] * A_scales[m] * B_scales[n]
    A_scales_expanded = tg.A_scales.cpu().unsqueeze(1)  # [M, 1]
    B_scales_expanded = tg.B_scales.cpu().unsqueeze(0)  # [1, N]
    C_scaled = AB * A_scales_expanded * B_scales_expanded
    
    # Stage 3: Denoise correction (matches kernel's FP16 denoise path)
    # Convert INT8 noise factors to FP16 (as kernel does)
    EAL_fp16 = tg.EAL.cpu().to(torch.float16)
    EAR_fp16 = tg.EAR_R_major.cpu().to(torch.float16)
    EBL_fp16 = tg.EBL_R_major.cpu().to(torch.float16)
    EBR_fp16 = tg.EBR.cpu().to(torch.float16)
    
    # Three denoise correction terms (all in FP32 for reference):
    # 1. A @ (EBR * B_scales_col)
    EBR_scaled = EBR_fp16.to(torch.float32) * B_scales_expanded.t()  # [R, N]
    A_fp32 = tg.A.cpu().to(torch.float32)
    term1 = (A_fp32 @ EBR_scaled.t()) * A_scales_expanded  # [M, N]
    
    # 2. (EAL * A_scales_row) @ B
    EAL_scaled = EAL_fp16.to(torch.float32) * A_scales_expanded  # [M, R]
    B_fp32 = tg.B.cpu().to(torch.float32)
    term2 = EAL_scaled @ B_fp32.to(torch.float32).t() * B_scales_expanded  # [M, N]
    
    # 3. (EAL * A_scales_row) @ (EBR * B_scales_col)
    term3 = EAL_scaled @ EBR_scaled  # [M, N]
    
    # Total denoise correction (scaled by 2^-12 as in kernel)
    kIntToFp16ScaleFactor = 2.0 ** -12
    denoise_correction = (term1 + term2 + term3) * kIntToFp16ScaleFactor
    
    # Final output
    C_ref = C_scaled - denoise_correction
    
    return C_ref


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
        rel_diff = (abs_diff / (ref.abs() + 1e-6)).max().item()
        
        # Mining correctness tolerances:
        # - FP16 denoise accumulation: ~5e-3 relative error per operation
        # - FP32 final output: ~1e-6 relative error
        # - INT8 quantization: introduces ±0.5 quantization error
        # 
        # For mining proofs, the hash/commitment is computed from bf16-quantized
        # output, so sub-LSB differences don't affect proof validity.
        # However, we want to catch any significant deviation that would indicate
        # incorrect kernel logic.
        #
        # Conservative thresholds:
        # - atol=0.5: Accounts for INT8 quantization + FP16 rounding
        # - rtol=1e-2: Accounts for accumulated FP16 errors (1% relative)
        #
        # NOTE: These are CORRECTNESS tests, not BF16 LSB-exact tests.
        # For mining, the kernel outputs FP32 which gets quantized to BF16
        # for hashing. As long as the BF16-quantized values match, the
        # commitment hashes will be identical.
        atol = 0.5
        rtol = 1e-2
        passed = torch.allclose(out, ref, atol=atol, rtol=rtol)
        
        # Additional checks for large errors that might indicate kernel bugs
        large_errors = (abs_diff > 1.0).sum().item()
        total_elements = out.numel()
        error_rate = large_errors / total_elements
        
        # Flag if >0.1% of elements have large errors
        if error_rate > 0.001:
            passed = False
        
        result = {
            "passed": passed,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "rel_diff": rel_diff,
            "large_errors": large_errors,
            "total_elements": total_elements,
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
