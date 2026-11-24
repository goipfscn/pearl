"""
Default compiled kernels configuration.

Minimal set of kernels for development and PR CI.
This replaces default_compiled_kernels.jsonnet with Python + pydantic.

NOTE: Architecture-specific configuration
- Hopper (sm_90a): Can use large tiles with stages=3 (up to 227KB SMEM)
- Blackwell (sm_120/121/100): Limited to 99KB SMEM per CTA, requires smaller tiles with stages=2
"""

import os
from pearl_gemm_build_utils.kernel_configs import (
    KernelCompilationGrid,
    MatmulKernelConfig,
    NoisingAKernelConfig,
    NoisingBKernelConfig,
)

# Detect target architecture
PEARL_GEMM_ARCH = os.getenv("PEARL_GEMM_ARCH", "sm_90a")
IS_BLACKWELL = PEARL_GEMM_ARCH in ("sm_100a", "sm_120a", "sm_121a")
IS_HOPPER = PEARL_GEMM_ARCH == "sm_90a"

# Build matmul kernels
_matmul_kernels = []

# 128x256x128 stages=3 — Hopper-only config (~210 KB SMEM)
# Exceeds Blackwell's 99KB SMEM limit; causes cudaFuncSetAttribute failure
if IS_HOPPER:
    for R in [64, 128]:
        _matmul_kernels.append(
            MatmulKernelConfig(
                tile_size_m=128,
                tile_size_n=256,
                tile_size_k=128,
                R=R,
                pipeline_stages=3,
                cM=1,
                cN=1,
            )
        )

# Blackwell-compatible tiles (all fit in 99KB SMEM with stages=2)
if IS_BLACKWELL:
    # 64x128x64 stages=2 — small efficient tile (~50KB SMEM)
    for R in [64, 128]:
        _matmul_kernels.append(
            MatmulKernelConfig(
                tile_size_m=64,
                tile_size_n=128,
                tile_size_k=64,
                R=R,
                pipeline_stages=2,
                cM=1,
                cN=1,
            )
        )

    # 128x128x64 stages=2 — balanced tile (~80KB SMEM)
    for R in [64, 128]:
        _matmul_kernels.append(
            MatmulKernelConfig(
                tile_size_m=128,
                tile_size_n=128,
                tile_size_k=64,
                R=R,
                pipeline_stages=2,
                cM=1,
                cN=1,
            )
        )

    # 64x64x64 stages=2 — minimum viable tile (comfortably under 99KB)
    for R in [64, 128]:
        _matmul_kernels.append(
            MatmulKernelConfig(
                tile_size_m=64,
                tile_size_n=64,
                tile_size_k=64,
                R=R,
                pipeline_stages=2,
                cM=1,
                cN=1,
            )
        )

# Noising A: 64x64, fp16/int32
_noising_a_kernels = [
    NoisingAKernelConfig(
        tile_size_m=64,
        tile_size_k=64,
        R=R,
        pipeline_stages=2,
        AxEBL_type=dtype,
    )
    for R in [64, 128]
    for dtype in ["fp16", "int32"]
]

# Noising B: 64x64, fp16/int32
_noising_b_kernels = [
    NoisingBKernelConfig(
        tile_size_n=64,
        tile_size_k=64,
        R=R,
        pipeline_stages=2,
        EARxBpEB_type=dtype,
    )
    for R in [64, 128]
    for dtype in ["fp16", "int32"]
]

KERNEL_CONFIGS = KernelCompilationGrid(
    matmul_kernels=_matmul_kernels,
    noising_a_kernels=_noising_a_kernels,
    noising_b_kernels=_noising_b_kernels,
)
