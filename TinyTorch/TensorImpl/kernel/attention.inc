#pragma once

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