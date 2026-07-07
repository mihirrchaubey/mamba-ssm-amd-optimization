#pragma once

#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

// Forward declaration of the actual GPU kernel launcher
// This function will be defined in src/selective_scan.hip
std::vector<at::Tensor> selective_scan_fwd_hip(
    const at::Tensor &u,      // Input sequence: (batch, dim, seqlen)
    const at::Tensor &delta,  // Time steps: (batch, dim, seqlen)
    const at::Tensor &A,      // State transition: (dim, dstate)
    const at::Tensor &B,      // Input projection: (batch, dstate, seqlen)
    const at::Tensor &C,      // Output projection: (batch, dstate, seqlen)
    const at::Tensor &D,      // Skip connection: (dim)
    const c10::optional<at::Tensor> &z, // Optional gating tensor
    const c10::optional<at::Tensor> &delta_bias, // Optional bias
    bool is_variable_B        // Flag for input-dependent B
);

// We will also need a backward pass function here eventually for training,
// but for the hackathon MVP, focusing on the forward inference pass is key.
