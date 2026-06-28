// AWQ INT4 dequant + gemm with shared memory tiling on input x.
// Naive baseline: each thread reads x[i] from global every iteration.
// SM version:   cooperatively load x tiles into shared memory, reuse across outputs.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__device__ constexpr int REORDER[8] = {0, 4, 1, 5, 2, 6, 3, 7};
constexpr int BK = 128;  // tile on K -> shared memory size

__global__ void awq_linear_sm_kernel(
    const nv_bfloat16* __restrict__ x,
    const int32_t*      __restrict__ qweight,
    const int32_t*      __restrict__ qzeros,
    const nv_bfloat16*  __restrict__ scales,
    nv_bfloat16*        __restrict__ out,
    const int batch, const int in_f, const int out_f, const int gs
) {
    __shared__ nv_bfloat16 sh_x[BK];
    constexpr int pf = 8;
    int num_o_blks = (out_f + blockDim.x - 1) / blockDim.x;
    int b = blockIdx.x / num_o_blks;
    int j = (blockIdx.x % num_o_blks) * blockDim.x + threadIdx.x;
    if (j >= out_f) return;

    const int col_idx = j / pf, nib = REORDER[j % pf], shift = nib * 4;
    float acc = 0.0f;

    for (int ks = 0; ks < in_f; ks += BK) {
        if (int k = threadIdx.x; k < BK && ks + k < in_f)
            sh_x[k] = x[b * in_f + ks + k];
        __syncthreads();
        int lim = min(BK, in_f - ks);
        for (int k = 0; k < lim; ++k) {
            int i = ks + k, grp = i / gs;
            int wp = qweight[i * (out_f / pf) + col_idx];
            int zp = qzeros[grp * (out_f / pf) + col_idx];
            float s = __bfloat162float(scales[grp * out_f + j]);
            int wv = (wp >> shift) & 0xF, zv = (zp >> shift) & 0xF;
            acc += __bfloat162float(sh_x[k]) * float(wv - zv) * s;
        }
        __syncthreads();
    }
    out[b * out_f + j] = __float2bfloat16(acc);
}

torch::Tensor awq_linear_forward_sm(
    torch::Tensor x, torch::Tensor qweight, torch::Tensor qzeros,
    torch::Tensor scales, int64_t group_size
) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16 && x.dim() == 2);
    auto batch = x.size(0), in_f = x.size(1), out_f = scales.size(1);
    auto out = torch::empty({batch, out_f}, x.options());
    constexpr int block = 256;
    int grid = batch * ((out_f + block - 1) / block);
    awq_linear_sm_kernel<<<grid, block>>>(
        (const nv_bfloat16*)x.data_ptr(), (const int32_t*)qweight.data_ptr(),
        (const int32_t*)qzeros.data_ptr(), (const nv_bfloat16*)scales.data_ptr(),
        (nv_bfloat16*)out.data_ptr(), batch, in_f, out_f, group_size);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &awq_linear_forward_sm, "AWQ dequant+gemm with SM tiling on x");
}
