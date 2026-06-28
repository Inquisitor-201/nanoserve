#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__device__ constexpr int REORDER[8] = {0, 4, 1, 5, 2, 6, 3, 7};

__global__ void awq_naive_kernel(
    const nv_bfloat16* __restrict__ x,
    const int32_t*      __restrict__ qweight,
    const int32_t*      __restrict__ qzeros,
    const nv_bfloat16*  __restrict__ scales,
    nv_bfloat16*        __restrict__ out,
    const int batch, const int in_f, const int out_f, const int gs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * out_f) return;
    int b = idx / out_f, j = idx % out_f;
    constexpr int pf = 8;
    int col_idx = j / pf, nib = REORDER[j % pf], shift = nib * 4;
    float acc = 0.0f;
    for (int i = 0; i < in_f; ++i) {
        int grp = i / gs;
        int wv = (qweight[i * (out_f / pf) + col_idx] >> shift) & 0xF;
        int zv = (qzeros[grp * (out_f / pf) + col_idx] >> shift) & 0xF;
        float xv = __bfloat162float(x[b * in_f + i]);
        float s = __bfloat162float(scales[grp * out_f + j]);
        acc += xv * float(wv - zv) * s;
    }
    out[idx] = __float2bfloat16(acc);
}

torch::Tensor forward(torch::Tensor x, torch::Tensor qw, torch::Tensor qz, torch::Tensor sc, int64_t gs) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16 && x.dim() == 2);
    auto out = torch::empty({x.size(0), sc.size(1)}, x.options());
    int grid = (x.size(0) * sc.size(1) + 255) / 256;
    awq_naive_kernel<<<grid, 256>>>(
        (const nv_bfloat16*)x.data_ptr(), (const int32_t*)qw.data_ptr(),
        (const int32_t*)qz.data_ptr(), (const nv_bfloat16*)sc.data_ptr(),
        (nv_bfloat16*)out.data_ptr(), x.size(0), x.size(1), sc.size(1), gs);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &forward, ""); }
