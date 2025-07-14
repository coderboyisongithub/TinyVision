#pragma once

__global__ void avgpool2d_backward_kernel(
    float* gradCol,
    const float* gradOut,
    int numel,
    int kSize,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float grad = gradOut[idx] * scale;
        for (int i = 0; i < kSize; ++i) {
            gradCol[idx * kSize + i] = grad;
        }
    }
}


