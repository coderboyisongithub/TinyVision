#pragma once

template<typename T>
__global__ void softmax_kernel(T* out, float inv_temperature,
                                       const T* __restrict__ inp, int N, int T) {
    constexpr float MAX_LOG_VALUE = 80.0f;
    float scale = min(inv_temperature, MAX_LOG_VALUE);

    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int row_idx = blockIdx.x;
    if(row_idx >= N * T) return;

    const T* x = inp + row_idx * T;
    T* row_out = out + row_idx * T;

    float thread_max = -FLT_MAX;
    float thread_sum = 0.0f;

    for (int i = threadIdx.x; i < T; i += blockDim.x) {
        float val = static_cast<float>(__ldg(x + i));
        thread_max = fmaxf(thread_max, val);
    }

    float maxval = cg::reduce(warp, thread_max, cg::greater<float>{});

    for (int i = threadIdx.x; i < T; i += blockDim.x) {
        float val = static_cast<float>(__ldg(x + i));
        thread_sum += expf(scale * (val - maxval));
    }

    float sumval = cg::reduce(warp, thread_sum, cg::plus<float>{});
    float norm = (sumval > 1e-7f) ? (1.f / sumval) : 0.f;

    for (int i = threadIdx.x; i < T; i += blockDim.x) {
        float val = static_cast<float>(__ldg(x + i));
        float ev = expf(scale * (val - maxval));
        row_out[i] = static_cast<T>(ev * norm);
    }
}

template<int BlockSize>
__global__ void softmax_backward_kernel(
    float* dpreatt,
    const float* datt,
    const float* att,
    int B, int T, int C, float scale)
{
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    __shared__ float block_sum[32];

    const int idx = blockIdx.z;
    const int row = blockIdx.y;

    const int offset = idx * T * T;
    const float* att_row = att + offset + row * T;
    const float* datt_row = datt + offset + row * T;
    float* dpreatt_row = dpreatt + offset + row * T;

    float dot_product = 0.0f;

    for (int col = threadIdx.x; col < T; col += blockDim.x) {
        dot_product += att_row[col] * datt_row[col];
    }

    dot_product = cg::reduce(warp, dot_product, cg::plus<float>{});

    if (warp.thread_rank() == 0) {
        block_sum[warp.meta_group_rank()] = dot_product;
    }
    block.sync();

    if (threadIdx.x < warp.size()) {
        dot_product = block_sum[threadIdx.x];
    } else {
        dot_product = 0.0f;
    }

    dot_product = cg::reduce(warp, dot_product, cg::plus<float>{});

    for (int col = threadIdx.x; col < T; col += blockDim.x) {
        float att_val = att_row[col];
        float datt_val = datt_row[col];

        // dP_i = att_i * (datt_i - Σ_j (att_j * datt_j))
        float dp_val = att_val * (datt_val - dot_product) * scale;
        dpreatt_row[col] = dp_val;
    }
}