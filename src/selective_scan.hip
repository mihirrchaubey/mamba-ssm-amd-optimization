#include "selective_scan.h"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

__device__ __forceinline__ float native_exp(float x) {
    return __expf(x);
}

__global__ void chunked_selective_scan_kernel(
    const float* __restrict__ u, const float* __restrict__ delta,
    const float* __restrict__ A, const float* __restrict__ B,
    const float* __restrict__ C, const float* __restrict__ D,
    const float* __restrict__ delta_bias, float* __restrict__ out,
    int batch, int dim, int seqlen, int dstate)
{
    int b = blockIdx.x;
    int d = threadIdx.x;
    if (b >= batch || d >= dim) return;

    extern __shared__ float s_A[];
    for (int n = 0; n < dstate; ++n) {
        s_A[d * dstate + n] = A[d * dstate + n];
    }
    __syncthreads();

    float h[16] = {0.0f};
    float dynamic_delta_bias = (delta_bias != nullptr) ? delta_bias[d] : 0.0f;
    float d_val = (D != nullptr) ? D[d] : 0.0f;

    int batch_stride = dim * seqlen;
    int channel_offset = b * batch_stride + d * seqlen;

    for (int t = 0; t < seqlen; ++t) {
        int idx = channel_offset + t;
        float u_val = u[idx];
        float dt_val = native_exp(delta[idx] + dynamic_delta_bias);

        for (int n = 0; n < dstate; ++n) {
            int b_idx = b * (dstate * seqlen) + n * seqlen + t;
            float dA = native_exp(dt_val * s_A[d * dstate + n]);
            float dB = dt_val * B[b_idx];
            h[n] = dA * h[n] + dB * u_val;
        }

        float y_val = 0.0f;
        for (int n = 0; n < dstate; ++n) {
            int c_idx = b * (dstate * seqlen) + n * seqlen + t;
            y_val += C[c_idx] * h[n];
        }
        y_val += d_val * u_val;
        out[idx] = y_val;
    }
}

std::vector<at::Tensor> selective_scan_fwd_hip(
    const at::Tensor &u, const at::Tensor &delta, const at::Tensor &A,
    const at::Tensor &B, const at::Tensor &C, const at::Tensor &D,
    const c10::optional<at::Tensor> &z, const c10::optional<at::Tensor> &delta_bias,
    bool is_variable_B)
{
    auto u_contig = u.contiguous();
    auto delta_contig = delta.contiguous();
    auto A_contig = A.contiguous();
    auto B_contig = B.contiguous();
    auto C_contig = C.contiguous();
    auto D_contig = D.defined() ? D.contiguous() : at::Tensor();
    auto bias_contig = delta_bias.has_value() ? delta_bias.value().contiguous() : at::Tensor();

    auto out = at::empty_like(u_contig);

    int batch = u_contig.size(0);
    int dim = u_contig.size(1);
    int seqlen = u_contig.size(2);
    int dstate = A_contig.size(1);

    int grid_size = batch;
    int block_size = dim;
    size_t shared_mem_bytes = dim * dstate * sizeof(float);

    hipLaunchKernelGGL(
        chunked_selective_scan_kernel, dim3(grid_size), dim3(block_size), shared_mem_bytes, 0,
        u_contig.data_ptr<float>(), delta_contig.data_ptr<float>(),
        A_contig.data_ptr<float>(), B_contig.data_ptr<float>(), C_contig.data_ptr<float>(),
        D.defined() ? D_contig.data_ptr<float>() : nullptr,
        delta_bias.has_value() ? bias_contig.data_ptr<float>() : nullptr,
        out.data_ptr<float>(), batch, dim, seqlen, dstate
    );
    return {out};
}
