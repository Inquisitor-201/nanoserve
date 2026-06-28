#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__device__ constexpr int REORDER[8] = {0, 4, 1, 5, 2, 6, 3, 7};
constexpr int BK = 128;

__global__ void awq_sm_kernel(
    const nv_bfloat16* __restrict__ x,
    const int32_t*      __restrict__ qweight,
    const int32_t*      __restrict__ qzeros,
    const nv_bfloat16*  __restrict__ scales,
    nv_bfloat16*        __restrict__ out,
    const int batch, const int in_f, const int out_f, const int gs
) {
    __shared__ nv_bfloat16 sh_x[BK];
    constexpr int pf = 8;
    constexpr int block = 256;
    int num_o_blocks = (out_f + block - 1) / block;
    int b = blockIdx.x / num_o_blocks;          // one batch row per block
    int j = (blockIdx.x % num_o_blocks) * block + threadIdx.x;
    int col_idx = j / pf, nib = REORDER[j % pf], shift = nib * 4;
    float acc = 0.0f;

    for (int ks = 0; ks < in_f; ks += BK) {
        // All threads in block load the same x[b, :] row (same b!)
        if (threadIdx.x < BK && ks + threadIdx.x < in_f)
            sh_x[threadIdx.x] = x[b * in_f + ks + threadIdx.x];
        __syncthreads();
        int lim = min(BK, in_f - ks);
        // Only threads with valid output column do compute
        if (j < out_f) {
            #pragma unroll
            for (int k = 0; k < lim; ++k) {
                int i = ks + k, grp = i / gs;
                int wv = (qweight[i * (out_f / pf) + col_idx] >> shift) & 0xF;
                int zv = (qzeros[grp * (out_f / pf) + col_idx] >> shift) & 0xF;
                float s = __bfloat162float(scales[grp * out_f + j]);
                acc += __bfloat162float(sh_x[k]) * float(wv - zv) * s;
            }
        }
        __syncthreads();
    }

    if (j < out_f)
        out[b * out_f + j] = __float2bfloat16(acc);
}

torch::Tensor forward(torch::Tensor x, torch::Tensor qw, torch::Tensor qz, torch::Tensor sc, int64_t gs) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16 && x.dim() == 2);
    auto out = torch::empty({x.size(0), sc.size(1)}, x.options());
    constexpr int block = 256;
    int num_o_blocks = (sc.size(1) + block - 1) / block;
    int grid = x.size(0) * num_o_blocks;
    awq_sm_kernel<<<grid, block>>>(
        (const nv_bfloat16*)x.data_ptr(), (const int32_t*)qw.data_ptr(),
        (const int32_t*)qz.data_ptr(), (const nv_bfloat16*)sc.data_ptr(),
        (nv_bfloat16*)out.data_ptr(), x.size(0), x.size(1), sc.size(1), gs);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &forward, ""); }
