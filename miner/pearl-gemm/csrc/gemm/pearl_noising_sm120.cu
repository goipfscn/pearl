// SM120 (Blackwell consumer / CC 12.0) noising fallback kernels
// Simple global-memory implementation - correctness baseline, not optimized
// Avoids TMA, WGMMA, ldmatrix, complex layouts

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace pearl {

// Kernel: Compute EA = EAL @ EAR^T, then ApEA = A + EA
// All operations in int8 with int32 accumulation where needed
__global__ void compute_ApEA_sm120(
    const int8_t* __restrict__ A,      // (m, k) row-major
    const int8_t* __restrict__ EAL,    // (m, R) row-major  
    const int8_t* __restrict__ EAR,    // (k, R) row-major (EAR_R_major)
    int8_t* __restrict__ ApEA,         // (m, k) row-major output
    int m, int k, int R,
    int stride_A_k,
    int stride_EAL_R,
    int stride_EAR_R,
    int stride_ApEA_k
) {
    // Each thread computes one output element ApEA[i, j]
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // m dimension
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // k dimension
    
    if (i < m && j < k) {
        // Compute EA[i,j] = sum_r EAL[i,r] * EAR[j,r]
        int32_t ea_accum = 0;
        for (int r = 0; r < R; ++r) {
            int32_t eal_val = static_cast<int32_t>(EAL[i * stride_EAL_R + r]);
            int32_t ear_val = static_cast<int32_t>(EAR[j * stride_EAR_R + r]);
            ea_accum += eal_val * ear_val;
        }
        
        // Convert to int8 (truncate/wrap)
        int8_t ea = static_cast<int8_t>(ea_accum);
        
        // Add A[i,j]
        int8_t a_val = A[i * stride_A_k + j];
        int8_t apea = a_val + ea;  // int8 addition with wrap
        
        ApEA[i * stride_ApEA_k + j] = apea;
    }
}

// Kernel: Compute AxEBL = A @ EBL with scale
__global__ void compute_AxEBL_sm120(
    const int8_t* __restrict__ A,      // (m, k)
    const int8_t* __restrict__ EBL,    // (R, k) (EBL_K_major, needs transpose)
    void* __restrict__ AxEBL_out,      // (m, R) - fp16 or int32
    int m, int k, int R,
    int stride_A_k,
    int stride_EBL_k,
    int stride_AxEBL_R,
    bool convert_to_fp16
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // m
    int r = blockIdx.y * blockDim.y + threadIdx.y;  // R
    
    if (i < m && r < R) {
        // AxEBL[i,r] = sum_k A[i,k] * EBL[r,k]
        int32_t accum = 0;
        for (int kk = 0; kk < k; ++kk) {
            int32_t a_val = static_cast<int32_t>(A[i * stride_A_k + kk]);
            int32_t ebl_val = static_cast<int32_t>(EBL[r * stride_EBL_k + kk]);
            accum += a_val * ebl_val;
        }
        
        if (convert_to_fp16) {
            // Scale by 2^-14 and convert to fp16
            float scaled = static_cast<float>(accum) * (1.0f / 16384.0f);
            __half result = __float2half(scaled);
            reinterpret_cast<__half*>(AxEBL_out)[i * stride_AxEBL_R + r] = result;
        } else {
            reinterpret_cast<int32_t*>(AxEBL_out)[i * stride_AxEBL_R + r] = accum;
        }
    }
}

// Kernel: Compute EB = EBR @ EBL^T, then BpEB = B + EB
__global__ void compute_BpEB_sm120(
    const int8_t* __restrict__ B,      // (n, k) row-major
    const int8_t* __restrict__ EBR,    // (n, R) row-major
    const int8_t* __restrict__ EBL,    // (k, R) row-major (EBL_R_major)
    int8_t* __restrict__ BpEB,         // (n, k) row-major output
    int n, int k, int R,
    int stride_B_k,
    int stride_EBR_R,
    int stride_EBL_R,
    int stride_BpEB_k
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // n
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // k
    
    if (i < n && j < k) {
        // Compute EB[i,j] = sum_r EBR[i,r] * EBL[j,r]
        int32_t eb_accum = 0;
        for (int r = 0; r < R; ++r) {
            int32_t ebr_val = static_cast<int32_t>(EBR[i * stride_EBR_R + r]);
            int32_t ebl_val = static_cast<int32_t>(EBL[j * stride_EBL_R + r]);
            eb_accum += ebr_val * ebl_val;
        }
        
        int8_t eb = static_cast<int8_t>(eb_accum);
        int8_t b_val = B[i * stride_B_k + j];
        int8_t bpeb = b_val + eb;
        
        BpEB[i * stride_BpEB_k + j] = bpeb;
    }
}

// Kernel: Compute EARxBpEB = BpEB @ EAR with scale
__global__ void compute_EARxBpEB_sm120(
    const int8_t* __restrict__ BpEB,   // (n, k)
    const int8_t* __restrict__ EAR,    // (R, k) (EAR_K_major, needs transpose)
    void* __restrict__ EARxBpEB_out,   // (n, R) - fp16 or int32
    int n, int k, int R,
    int stride_BpEB_k,
    int stride_EAR_k,
    int stride_EARxBpEB_R,
    bool convert_to_fp16
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // n
    int r = blockIdx.y * blockDim.y + threadIdx.y;  // R
    
    if (i < n && r < R) {
        // EARxBpEB[i,r] = sum_k BpEB[i,k] * EAR[r,k]
        int32_t accum = 0;
        for (int kk = 0; kk < k; ++kk) {
            int32_t bpeb_val = static_cast<int32_t>(BpEB[i * stride_BpEB_k + kk]);
            int32_t ear_val = static_cast<int32_t>(EAR[r * stride_EAR_k + kk]);
            accum += bpeb_val * ear_val;
        }
        
        if (convert_to_fp16) {
            // Scale by 2^-12
            float scaled = static_cast<float>(accum) * (1.0f / 4096.0f);
            __half result = __float2half(scaled);
            reinterpret_cast<__half*>(EARxBpEB_out)[i * stride_EARxBpEB_R + r] = result;
        } else {
            reinterpret_cast<int32_t*>(EARxBpEB_out)[i * stride_EARxBpEB_R + r] = accum;
        }
    }
}

// Host launcher for noise_A with SM120 fallback
void launch_noise_A_sm120(
    const int8_t* A, const int8_t* EAL, 
    void* AxEBL, int8_t* ApEA,
    const int8_t* EAR_R_major, const int8_t* EBL_K_major,
    int m, int k, int R,
    bool convert_AxEBL,
    cudaStream_t stream
) {
    // Assume row-major contiguous for simplicity
    int stride_A = k;
    int stride_EAL = R;
    int stride_EAR = R;  // EAR_R_major is (k, R)
    int stride_EBL = k;  // EBL_K_major is (R, k)
    int stride_ApEA = k;
    int stride_AxEBL = R;
    
    // Launch ApEA kernel
    dim3 block_apea(16, 16);
    dim3 grid_apea((m + 15) / 16, (k + 15) / 16);
    compute_ApEA_sm120<<<grid_apea, block_apea, 0, stream>>>(
        A, EAL, EAR_R_major, ApEA,
        m, k, R,
        stride_A, stride_EAL, stride_EAR, stride_ApEA
    );
    
    // Launch AxEBL kernel
    dim3 block_axebl(16, 16);
    dim3 grid_axebl((m + 15) / 16, (R + 15) / 16);
    compute_AxEBL_sm120<<<grid_axebl, block_axebl, 0, stream>>>(
        A, EBL_K_major, AxEBL,
        m, k, R,
        stride_A, stride_EBL, stride_AxEBL,
        convert_AxEBL
    );
}

// Host launcher for noise_B with SM120 fallback
void launch_noise_B_sm120(
    const int8_t* B, const int8_t* EBR,
    void* EARxBpEB, int8_t* BpEB,
    const int8_t* EAR_K_major, const int8_t* EBL_R_major,
    int n, int k, int R,
    bool convert_EARxBpEB,
    cudaStream_t stream
) {
    int stride_B = k;
    int stride_EBR = R;
    int stride_EAR = k;  // EAR_K_major is (R, k)
    int stride_EBL = R;  // EBL_R_major is (k, R)
    int stride_BpEB = k;
    int stride_EARxBpEB = R;
    
    // Launch BpEB kernel
    dim3 block_bpeb(16, 16);
    dim3 grid_bpeb((n + 15) / 16, (k + 15) / 16);
    compute_BpEB_sm120<<<grid_bpeb, block_bpeb, 0, stream>>>(
        B, EBR, EBL_R_major, BpEB,
        n, k, R,
        stride_B, stride_EBR, stride_EBL, stride_BpEB
    );
    
    // Launch EARxBpEB kernel (depends on BpEB)
    // No sync needed - kernels on same stream are ordered
    dim3 block_ear(16, 16);
    dim3 grid_ear((n + 15) / 16, (R + 15) / 16);
    compute_EARxBpEB_sm120<<<grid_ear, block_ear, 0, stream>>>(
        BpEB, EAR_K_major, EARxBpEB,
        n, k, R,
        stride_BpEB, stride_EAR, stride_EARxBpEB,
        convert_EARxBpEB
    );
}

}  // namespace pearl
