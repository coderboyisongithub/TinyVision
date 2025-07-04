#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <float.h>
namespace cg = cooperative_groups;
__device__ float& vec_at(float4& vec, int index) {
    return reinterpret_cast<float*>(&vec)[index];
}

__device__ float vec_at(const float4& vec, int index) {
    return reinterpret_cast<const float*>(&vec)[index];
}

template<class T>
__host__ __device__ T ceil_div(T dividend, T divisor) {
    return (dividend + divisor-1) / divisor;
}

__global__ void permute_kernel(float* q, float* k, float* v,
                               const float* inp,
                               int B, int N, int NH, int d) {
    // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, NH, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]

    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int inp_idx = \
            (b * N * 3 * NH * d)
            +   (n * 3 * NH * d)
            +       (0 * NH * d)
            +          (nh_ * d)
            +                d_;

        q[idx] = inp[inp_idx];
        k[idx] = inp[inp_idx + NH * d];
        v[idx] = inp[inp_idx + 2 * (NH * d)];
    }
}

__global__ void permute_kernel_backward(float* dinp,
                                        const float* dq, const float* dk, const float* dv,
                                        int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
        dinp[inp_idx] += dq[idx];
        dinp[inp_idx + NH * d] += dk[idx];
        dinp[inp_idx + 2 * (NH * d)] += dv[idx];
    }
}

__global__ void unpermute_kernel(const float* inp, float *out, int B, int N, int NH, int d) {
   // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        out[other_idx] = inp[idx];
    }
}

__global__ void unpermute_kernel_backward(float* dinp, const float *dout, int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        dinp[idx] += dout[other_idx];
    }
}
template<typename type>
__global__ void softmax_autoregressive_forward_kernel5(type* out, float inv_temperature, const type* inp, int N, int T) {
    // inp, out shape: (N, T, T), where N = B * NH
    // fuses the multiplication by scale inside attention
    // directly autoregressive, so we only compute the lower triangular part
    // uses the online softmax algorithm
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if(idx >= N * T) {
        return;
    }
    int own_pos = idx % T;
    int pos_by_4 = own_pos / 4;

    // one row of inp, i.e. inp[idx, :] of shape (T,)
    const type* x = inp + idx * T;

    // not INF, so we don't get NaNs accidentally when subtracting two values.
    float maxval = -FLT_MAX;
    float sumval = 0.0f;

    const float4* x_vec = reinterpret_cast<const float4*>(x);
    for (int i = warp.thread_rank(); i < pos_by_4; i += warp.size()) {
        float4 v = x_vec[i];
        float old_maxval = maxval;
        for(int k = 0; k < 4; ++k) {
            maxval = fmaxf(maxval, vec_at(v, k));
        }
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        for(int k = 0; k < 4; ++k) {
            sumval += expf(inv_temperature * (vec_at(v, k) - maxval));
        }
    }

    if(4*pos_by_4 + warp.thread_rank() <= own_pos) {
        float old_maxval = maxval;
        maxval = fmaxf(maxval, x[4*pos_by_4 + warp.thread_rank()]);
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        sumval += expf(inv_temperature * (x[4*pos_by_4 + warp.thread_rank()] - maxval));
    }

    float global_maxval = cg::reduce(warp, maxval, cg::greater<float>{});
    sumval *= expf(inv_temperature * (maxval - global_maxval));

    float sum = cg::reduce(warp, sumval, cg::plus<float>{});
    float norm = 1.f / sum;

    // divide the whole row by the sum
    for (int i = warp.thread_rank(); i <= own_pos; i += warp.size()) {
        // recalculation is faster than doing the round-trip through memory.
        float ev = expf(inv_temperature * (__ldcs(x + i) - global_maxval));
        __stcs(out + idx * T + i, ev * norm);
    }
}

template<int BlockSize>
__global__ void softmax_autoregressive_backward_kernel7(float* dpreatt, const float* datt, const float* att,
                                                        int B, int T, int C, float scale) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ float block_acc[32];

    int idx = blockIdx.y;
    int t = blockIdx.x;

    att += idx * T * T;
    datt += idx * T * T;
    dpreatt += idx * T * T;

    const float* att_bth = att + t * T;
    const float* datt_bth = datt + t * T;
    float* dpreatt_bth = dpreatt + t * T;

    if(warp.meta_group_rank() == 0) {
        block_acc[warp.thread_rank()] = 0;
    }

    float local_sum = 0;
    for(int t2 = block.thread_rank(); t2 <= t; t2 += BlockSize) {
        local_sum += att_bth[t2] * datt_bth[t2];
    }

    block_acc[warp.meta_group_rank()] = cg::reduce(warp, local_sum, cg::plus<float>{});
    block.sync();
    local_sum = cg::reduce(warp, block_acc[warp.thread_rank()], cg::plus<float>{});

    for (int t3 = block.thread_rank(); t3 <= t; t3 += BlockSize) {
        float acc = att_bth[t3] * (datt_bth[t3] - local_sum);
        dpreatt_bth[t3] = scale * acc;
    }
}

template<int BlockSize>
__global__ void softmax_noncausal_backward_kernel(
    float* dpreatt, const float* datt, const float* att,
    int B, int T, int C, float scale)
{
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ float block_acc[32];

    int batch_idx = blockIdx.y;
    int row_idx = blockIdx.x;

    const float* att_batch = att + batch_idx * T * T;
    const float* datt_batch = datt + batch_idx * T * T;
    float* dpreatt_batch = dpreatt + batch_idx * T * T;

    const float* att_row = att_batch + row_idx * T;
    const float* datt_row = datt_batch + row_idx * T;
    float* dpreatt_row = dpreatt_batch + row_idx * T;

    if (warp.meta_group_rank() == 0) {
        block_acc[warp.thread_rank()] = 0;
    }
    block.sync();

    float local_sum = 0;
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        local_sum += att_row[j] * datt_row[j];
    }

    float warp_sum = cg::reduce(warp, local_sum, cg::plus<float>{});

    if (warp.thread_rank() == 0) {
        block_acc[warp.meta_group_rank()] = warp_sum;
    }
    block.sync();

    float row_sum = 0;
    if (threadIdx.x < warp.meta_group_size()) {
        row_sum = block_acc[threadIdx.x];
    }
    row_sum = cg::reduce(warp, row_sum, cg::plus<float>{});

    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        float grad = att_row[j] * (datt_row[j] - row_sum);
        dpreatt_row[j] = scale * grad;
    }
}

__global__ void softmax_nocasual_forward_kernel(float* out, float scale, const float* inp, int N, int T) {
    // inp, out shape: (N, T, T)
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int row_idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if(row_idx >= N * T) return;
    const float* x = inp + row_idx * T;
    float maxval = -FLT_MAX;
    float sumval = 0.0f;
    for (int i = warp.thread_rank(); i < T; i += warp.size()) {
        float scaled_val = x[i] * scale;
        maxval = fmaxf(maxval, scaled_val);
    }
    float row_max = cg::reduce(warp, maxval, cg::greater<float>{});

    for (int i = warp.thread_rank(); i < T; i += warp.size()) {
        float val = expf(x[i] * scale - row_max);
        sumval += val;
    }
    float row_sum = cg::reduce(warp, sumval, cg::plus<float>{});

    for (int i = warp.thread_rank(); i < T; i += warp.size()) {
        float val = expf(x[i] * scale - row_max);
        out[row_idx * T + i] = val / row_sum;
    }
}

void launch_softmax_backward(float* dpreatt, float* datt, const float* att, int B, int T, int C, int NH, int block_size,int causal) {
    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf(hs);
    if (causal)
        softmax_autoregressive_backward_kernel7<512><<<dim3(T, B * NH), block_size>>>
                                                              (dpreatt, datt, att, B, T, C, scale);
    else
        softmax_noncausal_backward_kernel<512><<<dim3(T, B * NH), block_size>>>
                                                              (dpreatt, datt, att, B, T, C, scale);
}

void launch_softmax_forward(float *att, float*  preatt, int B, int T, int HS, int NH, int block_size,int causal) {
    float scale = 1.0 / sqrtf(HS);
    int grid_size = ceil_div(B * NH * T * 32, block_size);
    if (causal)
        softmax_autoregressive_forward_kernel5<<<grid_size,block_size>>>( att, scale,preatt, B * NH,  T);
    else
        softmax_nocasual_forward_kernel<<<grid_size,block_size>>>( att, scale,preatt, B * NH,  T);
}
