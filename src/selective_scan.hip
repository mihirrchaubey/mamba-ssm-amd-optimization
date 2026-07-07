#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <torch/extension.h>
#include <vector>

// Device function for fast exponentiation using hardware native instructions
__device__ __forceinline__ float native_exp(float x) {
    return __expf(x);
}

// Fused Forward Kernel
// Maps blocks to (Batch * Dim) and threads to parallel elements where applicable.
// For a simple baseline, each block processes one channel (B, D) sequentially across length L.
__global__ void selective_scan_fwd_kernel(
    const float* __restrict__ u,          // (B, D, L)
    const float* __restrict__ delta,      // (B, D, L)
    const float* __restrict__ A,          // (D, N)
    const float* __restrict__ B,          // (B, N, L)
    const float* __restrict__ C,          // (B, N, L)
    const float* __restrict__ D,          // (D)
    const float* __restrict__ delta_bias, // (D)
    float* __restrict__ out,              // (B, D, L)
    int batch, int dim, int seqlen, int dstate) 
{
    // Identify which Batch and Dimension this thread block is responsible for
    int b = blockIdx.x / dim;
    int d = blockIdx.x % dim;
    
    if (b >= batch || d >= dim) return;

    // Local dynamic thread indexes (for internal block execution)
    int tx = threadIdx.x;

    // Allocate Local Data Share (LDS) or registers for the hidden state vector h (size: N)
    // For optimization, keeping this intermediate state in fast registers minimizes HBM latency.
    extern __shared__ float shared_mem[]; // Can be expanded for tiled parallel scan
    
    // Dynamically tracking the hidden state vector h for this specific batch/dim channel
    // Allocate h array in registers per thread for sequential block tracking
    // Max dstate typically 16 or 32 for Mamba
    float h[32] = {0.0f}; 

    float dynamic_delta_bias = (delta_bias != nullptr) ? delta_bias[d] : 0.0f;
    float d_val = (D != nullptr) ? D[d] : 0.0f;

    // Stride calculations
    int batch_stride = dim * seqlen;
    int channel_offset = b * batch_stride + d * seqlen;

    // Loop over the sequence length sequentially within this baseline thread block
    for (int t = 0; t < seqlen; ++t) {
        int idx = channel_offset + t;
        
        // Read inputs
        float u_val = u[idx];
        float dt_val = native_exp(delta[idx] + dynamic_delta_bias); // Discretization step

        // Compute the recurrent update for each hidden state element up to dstate
        for (int n = 0; n < dstate; ++n) {
            // Address mappings for B and C matrices
            int b_idx = b * (dstate * seqlen) + n * seqlen + t;
            int c_idx = b * (dstate * seqlen) + n * seqlen + t;
            int a_idx = d * dstate + n;

            float a_val = A[a_idx];
            float b_val = B[b_idx];

            // Zero-Order Hold (ZOH) discretization mechanics
            float dA = native_exp(dt_val * a_val);
            float dB = dt_val * b_val;

            // Update state: h_t = dA * h_{t-1} + dB * u_t
            h[n] = dA * h[n] + dB * u_val;
        }

        // Project hidden state to output: y_t = C_t * h_t + D * u_t
        float y_val = 0.0f;
        for (int n = 0; n < dstate; ++n) {
            int c_idx = b * (dstate * seqlen) + n * seqlen + t;
            y_val += C[c_idx] * h[n];
        }
        y_val += d_val * u_val;

        // Write to global memory execution array
        out[idx] = y_val;
    }
}

// Host launcher function declared in include/selective_scan.h
std::vector<at::Tensor> selective_scan_fwd_hip(
    const at::Tensor &u,
    const at::Tensor &delta,
    const at::Tensor &A,
    const at::Tensor &B,
    const at::Tensor &C,
    const at::Tensor &D,
    const c10::optional<at::Tensor> &z,
    const c10::optional<at::Tensor> &delta_bias,
    bool is_variable_B) 
{
    // Ensure input tensors are contiguous in memory
    auto u_contig = u.contiguous();
    auto delta_contig = delta.contiguous();
    auto A_contig = A.contiguous();
    auto B_contig = B.contiguous();
    auto C_contig = C.contiguous();
    auto D_contig = D.has_value() ? D.value().contiguous() : at::Tensor();
    auto bias_contig = delta_bias.has_value() ? delta_bias.value().contiguous() : at::Tensor();

    auto out = at::empty_like(u_contig);

    int batch = u_contig.size(0);
    int dim = u_contig.size(1);
    int seqlen = u_contig.size(2);
    int dstate = A_contig.size(1);

    // Grid and block configurations
    // One block per independent channel (Batch * Dimension)
    int grid_size = batch * dim;
    int block_size = 64; // Can be adjusted when implementing chunked associative scans

    // Launch the HIP kernel
    hipLaunchKernelGGL(
        selective_scan_fwd_kernel,
        dim3(grid_size),
        dim3(block_size),
        0, 0, // Shared memory bytes, stream
        u_contig.data_ptr<float>(),
        delta_contig.data_ptr<float>(),
        A_contig.data_ptr<float>(),
        B_contig.data_ptr<float>(),
        C_contig.data_ptr<float>(),
        D.has_value() ? D_contig.data_ptr<float>() : nullptr,
        delta_bias.has_value() ? bias_contig.data_ptr<float>() : nullptr,
        out.data_ptr<float>(),
        batch, dim, seqlen, dstate
    );

    return {out};
}
