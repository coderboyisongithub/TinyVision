#pragma once

__global__ void computePrefixSumKernel(const float* mask, int* prefixSum, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        prefixSum[idx] = mask[idx] ? 1.0f : 0.0f;
    }
}

__global__ void gatherElementsKernel(const float* input, const int* prefixSum, float* out_indices, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int prev_prefix = (idx > 0) ? prefixSum[idx-1] : 0;
    if (prefixSum[idx] != prev_prefix) {
        int pos = prefixSum[idx] - 1;
        if (pos >= 0) {
            output[pos] = input[idx];
            out_indices[pos] = static_cast<float>(idx);
        }
    }
}

__global__ void scatterElementsKernel(float* original, const int* prefixSum, const float* processed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && prefixSum[idx] != prefixSum[idx - (idx>0)]) {
        int pos = prefixSum[idx] - 1;
        original[idx] = processed[pos];
    }
}

__global__ void maskedSelectKernel(
    const float* a,
    const float* mask,
    float* output,
    int32_t* indices,
    int* counter,
    int n
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n && mask[idx] != 0.0f) {
    int pos = atomicAdd(counter, 1);
    output[pos] = a[idx];
    indices[pos] = idx;
  }
}

__global__ void maskedSelectWarpKernel(
    const float* a,
    const float* mask,
    float* output,
    float* indices,
    int* globalCounter,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;

    __shared__ int warpOffsets[32]; // 假设最多 32 个 warp/块
    if (threadIdx.x < 32) warpOffsets[warpId] = 0;
    __syncthreads();

    unsigned validMask = __ballot_sync(0xFFFFFFFF, idx < n && mask[idx] != 0.0f);
    int validCount = __popc(validMask);

    if (laneId == 0) {
        warpOffsets[warpId] = atomicAdd(globalCounter, validCount);
    }
    __syncthreads();

    int warpBase = warpOffsets[warpId];
    int localIdx = __popc(validMask & ((1 << laneId) - 1));

    if (idx < n && mask[idx] != 0.0f) {
        int globalPos = warpBase + localIdx;
        output[globalPos] = a[idx];
        indices[globalPos] = idx;
    }
}

__global__ void scatterGradKernel(
    const float* grad_output,
    const float* indices,
    float* grad_input,
    int num_valid,
    int grad_input_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < num_valid; i += blockDim.x * gridDim.x) {
        int dst_pos = indices[i];
        if (dst_pos >= 0 && dst_pos < grad_input_size) {
            atomicAdd(&grad_input[dst_pos], grad_output[i]);
        }
    }
}