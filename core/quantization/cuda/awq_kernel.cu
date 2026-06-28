// Simple correct int4 AWQ dequant + gemm kernel (naive, one thread per output element).
// Uses AutoAWQ nibble reorder [0,4,1,5,2,6,3,7].

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// AutoAWQ packs 8 INT4 weights into one int32 in this order
__device__ constexpr int REORDER[8] = {0, 4, 1, 5, 2, 6, 3, 7};

__global__ void awq_linear_kernel(
    const nv_bfloat16* __restrict__ x,
    const int32_t*      __restrict__ qweight,
    const int32_t*      __restrict__ qzeros,
    const nv_bfloat16*  __restrict__ scales,
    nv_bfloat16*        __restrict__ out,
    const int batch,
    const int in_features,
    const int out_features,
    const int group_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * out_features) return;

    const int b = idx / out_features;
    const int j = idx % out_features;
    constexpr int pf = 8;

    float acc = 0.0f;

    for (int i = 0; i < in_features; ++i) {
        const int col_idx = j / pf;
        const int nib     = REORDER[j % pf];
        const int shift   = nib * 4;

        const int w_packed = qweight[i * (out_features / pf) + col_idx];
        const int w_val    = (w_packed >> shift) & 0xF;

        const int group    = i / group_size;
        const int z_packed = qzeros[group * (out_features / pf) + col_idx];
        const int z_val    = (z_packed >> shift) & 0xF;

        const float s = __bfloat162float(scales[group * out_features + j]);
        const float w = static_cast<float>(w_val - z_val) * s;
        const float xv = __bfloat162float(x[b * in_features + i]);
        acc += xv * w;
    }

    out[idx] = __float2bfloat16(acc);
}

torch::Tensor awq_linear_forward(
    torch::Tensor x, torch::Tensor qweight, torch::Tensor qzeros,
    torch::Tensor scales, int64_t group_size
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(x.dim() == 2, "x must be 2-D [batch, in_features]");

    const auto batch = x.size(0), in_f = x.size(1), out_f = scales.size(1);
    auto out = torch::empty({batch, out_f}, x.options());

    constexpr int block = 256;
    const int grid = (batch * out_f + block - 1) / block;
    awq_linear_kernel<<<grid, block>>>(
        (const nv_bfloat16*)x.data_ptr(),
        (const int32_t*)qweight.data_ptr(),
        (const int32_t*)qzeros.data_ptr(),
        (const nv_bfloat16*)scales.data_ptr(),
        (nv_bfloat16*)out.data_ptr(),
        batch, in_f, out_f, group_size
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "kernel failed");
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &awq_linear_forward,
          "Correct int4 AWQ dequant + gemm (nibble reorder)");
}
