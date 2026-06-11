// Naive fused int4 AWQ dequant + gemm kernel.
//
// One thread per output element.  Each thread loops over every input
// feature, unpacks int4 weight / zero from global memory, dequantises
// on the fly, and accumulates into the dot product.
//
// This is deliberately the simplest possible correct implementation.
// No shared memory, no tiling, no tensor cores — just a proof of
// correctness that the pipeline works end-to-end.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__global__ void awq_linear_naive_kernel(
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

    const int b = idx / out_features;           // batch index
    const int j = idx % out_features;           // output-feature column
    constexpr int pack_factor = 8;              // 32 bits / 4 bits per value

    float acc = 0.0f;

    for (int i = 0; i < in_features; ++i) {
        // --- unpack int4 weight value ---
        const int w_packed = qweight[i * (out_features / pack_factor) + j / pack_factor];
        const int w_shift  = (j % pack_factor) * 4;
        const int w_val    = (w_packed >> w_shift) & 0xF;

        // --- group and zero-point ---
        const int group    = i / group_size;
        const int z_packed = qzeros[group * (out_features / pack_factor) + j / pack_factor];
        const int z_val    = (z_packed >> w_shift) & 0xF;

        // --- scale (bf16 → float) ---
        const float s = __bfloat162float(scales[group * out_features + j]);

        // --- dequantized weight = (q - z) * s ---
        const float w = static_cast<float>(w_val - z_val) * s;

        // --- input × weight ---
        const float x_val = __bfloat162float(x[b * in_features + i]);
        acc += x_val * w;
    }

    out[idx] = __float2bfloat16(acc);
}


// ── C++ / PyBind11 wrapper ─────────────────────────────────────────

torch::Tensor awq_linear_forward(
    torch::Tensor x,
    torch::Tensor qweight,
    torch::Tensor qzeros,
    torch::Tensor scales,
    int64_t group_size
) {
    TORCH_CHECK(x.is_cuda(),       "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(x.dim() == 2,      "x must be 2-D [batch, in_features]");

    const auto batch        = x.size(0);
    const auto in_features  = x.size(1);
    const auto out_features = scales.size(1);
    const auto num_groups   = in_features / group_size;

    TORCH_CHECK(qweight.size(0) == in_features);
    TORCH_CHECK(qweight.size(1) == out_features / 8);
    TORCH_CHECK(qzeros.size(0)  == num_groups);
    TORCH_CHECK(qzeros.size(1)  == out_features / 8);
    TORCH_CHECK(scales.size(0)  == num_groups);
    TORCH_CHECK(scales.size(1)  == out_features);

    auto out = torch::empty({batch, out_features}, x.options());

    constexpr int block_size = 256;
    const int total_threads = batch * out_features;
    const int grid_size     = (total_threads + block_size - 1) / block_size;

    awq_linear_naive_kernel<<<grid_size, block_size>>>(
        reinterpret_cast<const nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const int32_t*>(qweight.data_ptr()),
        reinterpret_cast<const int32_t*>(qzeros.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(scales.data_ptr()),
        reinterpret_cast<nv_bfloat16*>(out.data_ptr()),
        batch, in_features, out_features, group_size
    );

    {
        const cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess,
                    "awq_linear_naive_kernel failed: ", cudaGetErrorString(err));
    }
    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &awq_linear_forward,
          "Naive fused int4 AWQ dequant + gemm (one thread per output element)");
}
