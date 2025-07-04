#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

__device__ void atomicAddX(__nv_bfloat16* addr, __nv_bfloat16 val) {
    uintptr_t ptr_val = reinterpret_cast<uintptr_t>(addr);
    __nv_bfloat162* ptr_bf16 = reinterpret_cast<__nv_bfloat162*>(ptr_val & ~uintptr_t(0x3));

    // Prepare the value to add, setting the other half to zero
    __nv_bfloat162 add_val = (ptr_val & 0x3) ? __halves2bfloat162(__ushort_as_bfloat16(0), val)
                                             : __halves2bfloat162(val, __ushort_as_bfloat16(0));
    atomicAdd(ptr_bf16, add_val);
}

__device__ void atomicAddX(half* addr, half val) {
    uintptr_t ptr_val = reinterpret_cast<uintptr_t>(addr);
    half2* ptr_fp16 = reinterpret_cast<half2*>(ptr_val & ~uintptr_t(0x3));

    // Prepare the value to add, setting the other half to zero
    half2 add_val = (ptr_val & 0x3) ? __halves2half2(__ushort_as_half(0), val)
                                    : __halves2half2(val, __ushort_as_half(0));
    atomicAdd(ptr_fp16, add_val);
}

__device__ void atomicAddX(float* addr, float val) {
    atomicAdd(addr, val);
}

__global__ void layernorm_forward_kernel(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const float*  __restrict__ inp, const float*  __restrict__ weight,
                                    const float* __restrict__ bias, int N, int C) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ float shared_sum[32]; // block_size max is 1024 = 32 * 32 warps
    __shared__ float shared_sum2[32]; // warps will be writing into shared memeory after warp-reduce
    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int idx = blockIdx.x; // simpoy one block per row
    // the row of input that this group of threads is responsible for
    const float* x = inp + idx * C;
    // thread coarsening through the row, reduce the sum in series
    float thread_sum = 0.0; // stores sum(x)
    float thread_sum2 = 0.0; // stores sum(x**2)
    // for (int i = C + threadIdx.x - blockDim.x; i >= 0; i -= blockDim.x) {
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float xi = x[i];
        thread_sum += xi;
        thread_sum2 += xi * xi;
    }
    // warp-level reduction
    float warp_sum = cg::reduce(warp, thread_sum, cg::plus<float>{}); // sum(x)
    float warp_sum2 = cg::reduce(warp, thread_sum2, cg::plus<float>{}); // sum(x**2)
    // store the warp-level reduction in shared memory (we could have lane_id == 0 guard but not needed)
    shared_sum[warp_id] = warp_sum;
    shared_sum2[warp_id] = warp_sum2;
    __syncthreads();
    // load results from shared memory to threads, pad with zeros for threads that are out of bounds
    warp_sum = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
    warp_sum2 = (lane_id < num_warps) ? shared_sum2[lane_id] : 0.0f;
    // now reduce the warp-level reductions
    float block_sum = cg::reduce(warp, warp_sum, cg::plus<float>{}); // sum(x)
    float block_sum2 = cg::reduce(warp, warp_sum2, cg::plus<float>{}); // sum(x**2)
    // mean, var, rstd
    block_sum /= C; // mean(x)
    block_sum2 /= C; // mean(x**2)
    float m = block_sum;
    float var = block_sum2 - m * m;
    float s = rsqrtf(var + 1e-5f);
    // store the mean, no need to cache it
    if(threadIdx.x == 0 && mean != nullptr) {
        __stcs(mean + idx, m);
    }
    // store the rstd, no need to cache it
    if(threadIdx.x == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }
    // final normalization and scaling by weight/bias
    float* o = out + idx * C;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float n = s * (__ldcs(x+i) - m);
        __stcs(o+i, n * weight[i] + bias[i]);
    }
}

template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
__global__ void layernorm_backward_kernel3(Tdinp* dinp, Tparams* dweight, Tparams* dbias,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C) {
    extern __shared__ float shared[]; // size = 2 * C

    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int base_idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    // init shared memory to zero
    #pragma unroll 4
    for(int i = threadIdx.x; i < C; i+= blockDim.x){
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    __syncthreads();

    int warps_in_grid = gridDim.x * warp.meta_group_size();
    for (int idx = base_idx; idx < B * T; idx += warps_in_grid) {
        int b = idx / T;
        int t = idx % T;

        const Tdout* dout_bt = dout + b * T * C + t * C;
        const Trest* inp_bt = inp + b * T * C + t * C;
        Tdinp* dinp_bt = dinp + b * T * C + t * C;
        const float mean_bt = (float)mean[b * T + t];
        const float rstd_bt = (float)rstd[b * T + t];

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = warp.thread_rank(); i < C; i  += warp.size()) {
            float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * (float)dout_bt[i];
            dnorm_mean += dnorm_i;
            dnorm_norm_mean += dnorm_i * norm_bti;
        }
        dnorm_mean = cg::reduce(warp, dnorm_mean, cg::plus<float>{});
        dnorm_norm_mean = cg::reduce(warp, dnorm_norm_mean, cg::plus<float>{});
        dnorm_mean = dnorm_mean / C;
        dnorm_norm_mean = dnorm_norm_mean / C;

        // now iterate again and accumulate all the gradients
        for (int i = warp.thread_rank(); i < C; i += warp.size()) {
            float dout_i = (float)__ldcs(&dout_bt[i]);
            float norm_bti = ((float)__ldcs(&inp_bt[i]) - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * dout_i;
            // gradient contribution to bias
            atomicAdd(&dbias_shared[i], dout_i);
            // gradient contribution to weight
            atomicAdd(&dweight_shared[i], norm_bti * dout_i);
            // gradient contribution to input
            float dval = 0.0f;
            dval += dnorm_i; // term 1
            dval -= dnorm_mean; // term 2
            dval -= norm_bti * dnorm_norm_mean; // term 3
            dval *= rstd_bt; // final scale
            dinp_bt[i] = (Tdinp)((float)dinp_bt[i] + dval);
        }
    }
    __syncthreads();

    for(int i = threadIdx.x; i < C; i+= blockDim.x) {
        atomicAddX(&dbias[i], (Tparams)dbias_shared[i]);
        atomicAddX(&dweight[i], (Tparams)dweight_shared[i]);
    }
}