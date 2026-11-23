#!/usr/bin/env python3
"""
Minimal sm_120 test - no external deps (miner_base, pearl_gateway, etc.)
Tests ONLY the CUDA kernel execution path.
"""
import sys
import torch

def test_basic_import():
    """Test 1: Can we import the extension?"""
    print("Test 1: Import pearl_gemm_cuda")
    try:
        import pearl_gemm_cuda
        print("  ✓ PASS: pearl_gemm_cuda imported")
        return True
    except ImportError as e:
        print(f"  ✗ FAIL: {e}")
        return False

def test_device():
    """Test 2: CUDA device check."""
    print("\nTest 2: CUDA device")
    if not torch.cuda.is_available():
        print("  ✗ FAIL: CUDA not available")
        return False
    
    name = torch.cuda.get_device_name()
    cc = torch.cuda.get_device_capability()
    print(f"  Device: {name}")
    print(f"  Compute capability: sm_{cc[0]}{cc[1]}")
    
    if cc[0] != 12 or cc[1] not in [0, 1]:
        print(f"  ⚠ WARNING: Expected sm_120/121, got sm_{cc[0]}{cc[1]}")
    else:
        print("  ✓ PASS: Correct architecture")
    return True

def test_simple_gemm():
    """Test 3: Simplest possible GEMM launch."""
    print("\nTest 3: Simple GEMM kernel launch")
    try:
        from pearl_gemm.test_components import noisy_gemm, GemmTensorGenerator, GEMMParam
        from pearl_gemm_build_utils.kernel_configs import KERNEL_CONFIGS
        
        # Use smallest compiled config
        configs = sorted(KERNEL_CONFIGS.matmul_kernels, 
                        key=lambda c: c.tile_size_m * c.tile_size_n)
        if not configs:
            print("  ✗ FAIL: No configs compiled")
            return False
        
        mc = configs[0]  # Smallest config
        print(f"  Using config: {mc.tile_size_m}x{mc.tile_size_n}x{mc.tile_size_k} R={mc.R} stages={mc.pipeline_stages}")
        
        # Minimal problem size
        gp = GEMMParam(
            m=128, n=256, k=256,  # Small size
            skip_noising_a=False,
            skip_noising_b=False,
            EARxBpEB_type_noising="fp16",
            AxEBL_type_noising="fp16",
            matmul_config=mc,
            use_variable_scales=True,
            skip_reduction=True,  # Skip PoW for basic test
        )
        
        tg = GemmTensorGenerator(gp)
        tg.generate()
        
        print("  Launching kernel...")
        noisy_gemm(
            A=tg.A, B=tg.B,
            EAL=tg.EAL, EAL_fp16=tg.EAL_fp16,
            EAR_R_major=tg.EAR_R_major,
            EBL_R_major=tg.EBL_R_major,
            EAR_K_major=tg.EAR_K_major,
            EBL_K_major=tg.EBL_K_major,
            EBR=tg.EBR, EBR_fp16=tg.EBR_fp16,
            AxEBL_fp16=tg.AxEBL_fp16,
            EARxBpEB_fp16=tg.EARxBpEB_fp16,
            ApEA=tg.ApEA, BpEB=tg.BpEB,
            A_scales=tg.A_scales, B_scales=tg.B_scales,
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
        
        print("  ✓ PASS: Kernel launched successfully")
        return True
        
    except RuntimeError as e:
        error_str = str(e)
        print(f"  ✗ FAIL: RuntimeError: {error_str}")
        
        if "wgmma" in error_str.lower() or "CUTE_ARCH" in error_str:
            print("\n  DIAGNOSIS: Hopper-only operation called on sm_120")
            print("  This means architecture guards are incomplete.")
            print("  Check for unguarded warpgroup_* calls in:")
            print("    - collective_mainloop.hpp")
            print("    - collective_epilogue.hpp")
            print("    - pearl_noisingA_kernel.h")
            print("    - pearl_noisingB_kernel.h")
            print("    - pearl_gemm_kernel.h")
            print("    - pow_utils.hpp")
        
        return False
    
    except Exception as e:
        print(f"  ✗ FAIL: {type(e).__name__}: {e}")
        return False

def main():
    print("=" * 70)
    print("Pearl sm_120 Minimal Kernel Test")
    print("=" * 70)
    print("Tests ONLY kernel launch, no external mining deps")
    print()
    
    results = []
    results.append(test_basic_import())
    results.append(test_device())
    results.append(test_simple_gemm())
    
    print("\n" + "=" * 70)
    if all(results):
        print("✓ ALL TESTS PASSED")
        print("Next: Run test_mining_proof_validation.sh for mining correctness")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        failed = sum(1 for r in results if not r)
        print(f"Failed: {failed}/{len(results)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
