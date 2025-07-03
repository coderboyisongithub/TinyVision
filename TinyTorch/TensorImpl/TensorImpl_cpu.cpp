/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "TensorImpl_cpu.h"

#include <algorithm>
#include <cassert>
#include <numeric>

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include "cblas.h"
#endif
#endif

#include "TensorImpl_cpu.inc"

namespace TinyTorch {

std::optional<unsigned long> RandomGeneratorCPU::seed_;
static std::random_device _r;
std::default_random_engine RandomGeneratorCPU::randomEngine_(_r());

void AllocatorCPU::allocate(void** ptr, size_t size) {
  if (pinned_) {
    *ptr = allocatePinned(size);
  } else {
    *ptr = allocateAlign(size, TENSOR_MEM_ALIGN);
  }
}

void AllocatorCPU::deallocate(void* ptr) {
  if (pinned_) {
    deallocatePinned(ptr);
  } else {
    deallocateAlign(ptr);
  }
}

void* AllocatorCPU::allocateAlign(size_t size, size_t alignment) {
#if !defined(_MSC_VER)
  return std::aligned_alloc(alignment, size);
#else
  return _aligned_malloc(size, alignment);
#endif
}

void AllocatorCPU::deallocateAlign(void* ptr) {
#if !defined(_MSC_VER)
  std::free(ptr);
#else
  _aligned_free(ptr);
#endif
}

#ifndef USE_CUDA
void* AllocatorCPU::allocatePinned(size_t size) { return nullptr; }

void AllocatorCPU::deallocatePinned(void* ptr) {}
#endif

TensorOpsCPU::TensorOpsCPU() {
  allocator_.setBaseAllocator(std::make_shared<AllocatorCPU>());
}

TensorOpsCPU::~TensorOpsCPU() { allocator_.clear(); }

template <typename OP>
void TensorOpsCPU::opSingle_(TensorImpl& t) {
  const OP opFunc;
  for (int32_t i = 0; i < t.elemCount_; i++) {
    opFunc(t.data_[i]);
  }
}

template <typename OP>
TensorImpl TensorOpsCPU::opSingle(const TensorImpl& t) {
  const OP opFunc;
  auto result = TensorImpl::shape(t.shape(), t.device_);
  for (int32_t i = 0; i < t.elemCount_; i++) {
    result.data_[i] = opFunc(t.data_[i]);
  }
  return result;
}

template <typename OP>
TensorImpl TensorOpsCPU::opPair(const TensorImpl& a, const TensorImpl& b) {
  const OP opFunc;
  auto result = TensorImpl::shape(a.shape(), a.device_);
  for (int32_t i = 0; i < a.elemCount_; i++) {
    result.data_[i] = opFunc(a.data_[i], b.data_[i]);
  }
  return result;
}

template <typename OP>
TensorImpl TensorOpsCPU::opPair(const TensorImpl& a, float b) {
  const OP opFunc;
  auto result = TensorImpl::shape(a.shape(), a.device_);
  for (int32_t i = 0; i < a.elemCount_; i++) {
    result.data_[i] = opFunc(a.data_[i], b);
  }
  return result;
}

template <typename OP>
TensorImpl TensorOpsCPU::opPair(float a, const TensorImpl& b) {
  const OP opFunc;
  auto result = TensorImpl::shape(b.shape(), b.device_);
  for (int32_t i = 0; i < b.elemCount_; i++) {
    result.data_[i] = opFunc(a, b.data_[i]);
  }
  return result;
}

template <typename OP>
void TensorOpsCPU::opPair_(TensorImpl& t, float b) {
  const OP opFunc;
  for (int32_t i = 0; i < t.elemCount_; i++) {
    t.data_[i] = opFunc(t.data_[i], b);
  }
}

template <typename OP>
void TensorOpsCPU::opPair_(TensorImpl& t, const TensorImpl& b) {
  const OP opFunc;
  for (int32_t i = 0; i < t.elemCount_; i++) {
    t.data_[i] = opFunc(t.data_[i], b.data_[i]);
  }
}

template <typename OP, bool REVERSE>
void TensorOpsCPU::broadcastImplLeadingOnes(TensorImpl& result,
                                            const TensorImpl& larger,
                                            const TensorImpl& smaller) {
  const OP opFunc;
  const int32_t n = smaller.elemCount_;
  for (int32_t idx = 0; idx < result.elemCount_; idx++) {
    auto& dataA = larger.data_[idx];
    auto& dataB = smaller.data_[idx % n];
    auto& dataRet = result.data_[idx];
    dataRet = REVERSE ? opFunc(dataB, dataA) : opFunc(dataA, dataB);
  }
}

template <typename OP, bool REVERSE>
void TensorOpsCPU::broadcastImplTrailingOnes(TensorImpl& result,
                                             const TensorImpl& larger,
                                             const TensorImpl& smaller) {
  const OP opFunc;
  const int32_t n = result.elemCount_ / smaller.elemCount_;
  for (int32_t idx = 0; idx < result.elemCount_; idx++) {
    auto& dataA = larger.data_[idx];
    auto& dataB = smaller.data_[idx / n];
    auto& dataRet = result.data_[idx];
    dataRet = REVERSE ? opFunc(dataB, dataA) : opFunc(dataA, dataB);
  }
}

template <typename OP>
void TensorOpsCPU::broadcastImplCommon(TensorImpl& result, const TensorImpl& a,
                                       const TensorImpl& b) {
  const OP opFunc;

  static int32_t cIndices[TENSOR_MAX_DIMS];
  static int32_t aIndices[TENSOR_MAX_DIMS];
  static int32_t bIndices[TENSOR_MAX_DIMS];
  memset(aIndices, 0, TENSOR_MAX_DIMS * sizeof(int32_t));
  memset(bIndices, 0, TENSOR_MAX_DIMS * sizeof(int32_t));

  for (int32_t i = 0; i < result.elemCount_; i++) {
    offsetToIndices(cIndices, result.shape_, i);
    for (int32_t j = 0; j < result.dimCount_; j++) {
      if (j >= result.dimCount_ - a.dimCount_) {
        int32_t aIndex = j - (result.dimCount_ - a.dimCount_);
        aIndices[aIndex] = (a.shape_[aIndex] != 1) ? cIndices[j] : 0;
      }
      if (j >= result.dimCount_ - b.dimCount_) {
        int32_t bIndex = j - (result.dimCount_ - b.dimCount_);
        bIndices[bIndex] = (b.shape_[bIndex] != 1) ? cIndices[j] : 0;
      }
    }
    auto aIdx = indicesToOffset(a.strides_, aIndices);
    auto bIdx = indicesToOffset(b.strides_, bIndices);
    result.data_[i] = opFunc(a.data_[aIdx], b.data_[bIdx]);
  }
}

template <typename OP>
void TensorOpsCPU::broadcastImpl(TensorImpl& result, const TensorImpl& a,
                                 const TensorImpl& b) {
  // fast broadcast with a
  if (b.elemCount_ == result.elemCount_) {
    if (isLeadingOnes(a.shape())) {
      broadcastImplLeadingOnes<OP, true>(result, b, a);
      return;
    }

    if (isTrailingOnes(a.shape())) {
      broadcastImplTrailingOnes<OP, true>(result, b, a);
      return;
    }
  }

  // fast broadcast with b
  if (a.elemCount_ == result.elemCount_) {
    if (isLeadingOnes(b.shape())) {
      broadcastImplLeadingOnes<OP, false>(result, a, b);
      return;
    }

    if (isTrailingOnes(b.shape())) {
      broadcastImplTrailingOnes<OP, false>(result, a, b);
      return;
    }
  }

  broadcastImplCommon<OP>(result, a, b);
}

template <typename OP>
TensorImpl TensorOpsCPU::opPairBroadcast(const TensorImpl& a,
                                         const TensorImpl& b) {
  Shape retShape;
  auto comp = checkShapeCompatible(a.shape(), b.shape(), retShape);
  if (comp == ShapeCompatible_Error) {
    error(__FUNCTION__, TensorError_ShapeNotAligned);
    return {};
  }
  if (comp == ShapeCompatible_SameShape) {
    return opPair<OP>(a, b);
  }

  auto result = TensorImpl::shape(retShape, a.device_);
  broadcastImpl<OP>(result, a, b);
  return result;
}

template <typename OP>
void TensorOpsCPU::opPairBroadcast_(TensorImpl& a, const TensorImpl& b) {
  Shape retShape;
  auto comp = checkShapeCompatible(a.shape(), b.shape(), retShape);
  if (comp == ShapeCompatible_Error) {
    error(__FUNCTION__, TensorError_ShapeNotAligned);
    return;
  }
  if (comp == ShapeCompatible_SameShape) {
    opPair_<OP>(a, b);
    return;
  }

  auto result = TensorImpl::shape(retShape, a.device_);
  broadcastImpl<OP>(result, a, b);
  a = std::move(result);
}

int32_t TensorOpsCPU::getReduceSrcIndex(const TensorImpl& ret,
                                        const TensorImpl& t, int32_t idx,
                                        int32_t dim, bool keepDims) {
  int32_t outIndex = idx;
  int32_t inIndex = 0;
  for (int32_t d = ret.dimCount_ - 1; d >= 0; d--) {
    int32_t coord = outIndex % ret.shape_[d];
    outIndex /= ret.shape_[d];
    if (keepDims || d < dim) {
      inIndex += coord * t.strides_[d];
    } else {
      inIndex += coord * t.strides_[d + 1];
    }
  }
  return inIndex;
}

int32_t TensorOpsCPU::getReduceDstIndex(const TensorImpl& t, int32_t idx,
                                        int32_t dim) {
  int32_t retIdx = 0;
  int32_t stride = 1;
  for (int32_t d = t.dimCount_ - 1; d >= 0; d--) {
    if (d != dim) {
      retIdx += (idx / t.strides_[d] % t.shape_[d]) * stride;
      stride *= t.shape_[d];
    }
  }
  return retIdx;
}

int32_t TensorOpsCPU::getReduceDstIndex(const TensorImpl& t, int32_t idx,
                                        const FixedVector<uint8_t>& inAxis) {
  int32_t retIdx = 0;
  int32_t stride = 1;
  for (int32_t d = t.dimCount_ - 1; d >= 0; d--) {
    if (0 == inAxis.data[d]) {
      retIdx += (idx / t.strides_[d] % t.shape_[d]) * stride;
      stride *= t.shape_[d];
    }
  }
  return retIdx;
}

template <typename OP>
void TensorOpsCPU::reduceAll(float* output, const float* input, int32_t n) {
  const OP op;
  float val = OP::defaultVal();
  for (int32_t i = 0; i < n; i++) {
    val = op(input[i], val);
  }
  *output = val;
}

template <typename OP>
void TensorOpsCPU::reduceAllIdx(float* output, const float* input, int32_t n) {
  const OP op;
  float val = OP::defaultVal();
  int32_t valIdx = 0;
  for (int32_t i = 0; i < n; i++) {
    if (op(val, input[i]) != val) {
      val = input[i];
      valIdx = i;
    }
  }
  *output = static_cast<float>(valIdx);
}

template <typename Compare, bool IsLastDim>
void TensorOpsCPU::reduceDimImpl(TensorImpl& values, TensorImpl& indices,
                                 const TensorImpl& t, int32_t dim,
                                 bool keepDims, float initVal, Compare comp) {
  const auto dimSize = t.shape_[dim];
  const auto stride = IsLastDim ? 1 : t.strides_[dim];

  for (int32_t i = 0; i < values.elemCount_; i++) {
    auto targetVal = initVal;
    int32_t targetIdx = 0;
    int32_t srcIdx = IsLastDim ? i * dimSize
                               : getReduceSrcIndex(values, t, i, dim, keepDims);
    for (int32_t j = 0; j < dimSize; j++) {
      auto val = t.data_[srcIdx];
      srcIdx += stride;
      if (comp(val, targetVal)) {
        targetVal = val;
        targetIdx = j;
      }
    }
    values.data_[i] = targetVal;
    indices.data_[i] = static_cast<float>(targetIdx);
  }
}

template <typename Compare>
std::pair<TensorImpl, TensorImpl> TensorOpsCPU::reduceDim(const TensorImpl& t,
                                                          int32_t dim,
                                                          bool keepDims,
                                                          float initVal,
                                                          Compare comp) {
  if (dim < 0) {
    dim += t.dimCount_;
  }
  if (dim < 0 || dim >= t.dimCount_) {
    error(__FUNCTION__, TensorError_InvalidAxis);
    return {};
  }

  const auto retShape = getReduceShape(t, dim, keepDims);
  auto values = TensorImpl::shape(retShape, t.device_);
  auto indices = TensorImpl::shape(retShape, t.device_);

  if (dim == t.dimCount_ - 1) {
    reduceDimImpl<Compare, true>(values, indices, t, dim, keepDims, initVal,
                                 comp);
  } else {
    reduceDimImpl<Compare, false>(values, indices, t, dim, keepDims, initVal,
                                  comp);
  }
  return {values, indices};
}



template <typename Op>
TensorImpl TensorOpsCPU::reduceMultiDim(const TensorImpl& t,
                                        const std::vector<int32_t>& dims,
                                        bool keepDims, Op op) {
  FixedVector<uint8_t> inAxis{};
  for (int32_t d : dims) {
    if (d < 0) {
      d += t.dimCount_;
    }
    if (d < 0 || d >= t.dimCount_) {
      error(__FUNCTION__, TensorError_InvalidAxis);
      return {};
    }
    inAxis.data[d] = 1;
  }

  const auto retShape = getReduceShape(t, inAxis, keepDims);
  auto ret = TensorImpl::shape(retShape, t.device_);

  fillConstant_(ret, 0);
  for (int32_t srcIdx = 0; srcIdx < t.elemCount_; srcIdx++) {
    int32_t retIdx = getReduceDstIndex(t, srcIdx, inAxis);
    op(ret, t, retIdx, srcIdx);
  }
  return ret;
}

void TensorOpsCPU::getSubIndices(
    int32_t* subIndices, const TensorImpl& t,
    const std::vector<std::reference_wrapper<TensorImpl>>& indices,
    int32_t idx) {
  for (int32_t i = 0; i < indices.size(); i++) {
    auto ind = (int32_t)indices[i].get().data_[idx];
    subIndices[i] = ind >= 0 ? ind : ind + t.shape_[i];
  }
}

void TensorOpsCPU::allocate(void** ptr, size_t size) {
  allocator_.allocate(ptr, size);
}

void TensorOpsCPU::deallocate(void* ptr) { allocator_.deallocate(ptr); }

void TensorOpsCPU::copyHostToDevice(void* dst, const void* src, size_t count) {
  std::memcpy(dst, src, count);
}

void TensorOpsCPU::copyOnDevice(void* dst, const void* src, size_t count) {
  std::memcpy(dst, src, count);
}

void TensorOpsCPU::copyDeviceToHost(void* dst, const void* src, size_t count) {
  std::memcpy(dst, src, count);
}

void TensorOpsCPU::fillConstant_(float* dst, float val, size_t count, Dtype T) {
  std::fill(dst, dst + count, val);
}

void TensorOpsCPU::fillConstant_(TensorImpl& t, float val) {
  std::fill(t.data_, t.data_ + t.elemCount_, val);
}

void TensorOpsCPU::fillLinSpace_(float* dst, float start, float step,
                                 size_t count) {
  for (size_t i = 0; i < count; i++) {
    dst[i] = start + (float)i * step;
  }
}

void TensorOpsCPU::fillRandUniform_(TensorImpl& t, float min, float max) {
  auto generator = RandomGeneratorCPU::getGenerator();
  std::uniform_real_distribution distribution(min, max);
  for (int32_t i = 0; i < t.elemCount_; i++) {
    t.data_[i] = distribution(generator);
  }
}

void TensorOpsCPU::fillRandNormal_(TensorImpl& t) {
  auto generator = RandomGeneratorCPU::getGenerator();
  std::normal_distribution distribution(0.0f, 1.0f);
  for (int32_t i = 0; i < t.elemCount_; i++) {
    t.data_[i] = distribution(generator);
  }
}

void TensorOpsCPU::fillRandBernoulli_(TensorImpl& t, float p) {
  auto generator = RandomGeneratorCPU::getGenerator();
  std::bernoulli_distribution distribution(p);
  for (int32_t i = 0; i < t.elemCount_; i++) {
    t.data_[i] = distribution(generator);
  }
}

TensorImpl attention_forward_qkv(TensorImpl& inp, TensorImpl& vaccum, TensorImpl& qkvr,
                                 TensorImpl& att, TensorImpl& preatt, int32_t NH) {
    // input is (B, T, 3C) Q,K,V
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int B = inp.shape()[0];
    int T = inp.shape()[1];
    int C = inp.shape()[2] / 3;
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);
    auto ret = TensorImpl::zerosLike(inp);
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                const float* query_t = inp.data() + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt.data() + b*NH*T*T + h*T*T + t*T;
                float* att_bth = att.data() + b*NH*T*T + h*T*T + t*T;
                // pass 1: calculate query dot key and maxval
                float maxval = -10000.0f; // TODO something better
                for (int t2 = 0; t2 <= t; t2++) {
                    const float* key_t2 = inp.data() + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    // (query_t) dot (key_t2)
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval) {
                        maxval = val;
                    }
                    preatt_bth[t2] = val;
                }
                // pad with -INFINITY outside of autoregressive region for debugging comparisons
                for (int t2 = t+1; t2 < T; t2++) {
                    preatt_bth[t2] = -INFINITY;
                }

                // pass 2: calculate the exp and keep track of sum
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                float* out_bth = ret.data() + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                for (int t2 = 0; t2 <= t; t2++) {
                    const float* value_t2 = inp.data() + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
    return ret;
}

std::vector<TensorImpl> attention_backward_qkv(const TensorImpl& dout,TensorImpl& inp,
                                               TensorImpl& qkvr, TensorImpl& vaccum, TensorImpl& att, int32_t NH) {
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    TensorImpl datt = TensorImpl::zerosLike(att);
    TensorImpl dpreatt = TensorImpl::zerosLike(att);
    TensorImpl dinp = TensorImpl::zerosLike(inp);
    int B = dout.shape()[0];
    int T = dout.shape()[1];
    int C = dout.shape()[2];
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float* att_bth = att.data() + b*NH*T*T + h*T*T + t*T;
                float* datt_bth = datt.data() + b*NH*T*T + h*T*T + t*T;
                float* dpreatt_bth = dpreatt.data() + b*NH*T*T + h*T*T + t*T;
                float* dquery_t = dinp.data() + b * T * C3 + t * C3 + h * hs;
                float* query_t = inp.data() + b * T * C3 + t * C3 + h * hs;

                // backward pass 4, through the value accumulation
                const float* dout_bth = dout.data() + b * T * C + t * C + h * hs;
                for (int t2 = 0; t2 < T; t2++) { // ADJUSTED! this was t2 <= t (see note on function)
                    float* value_t2 = inp.data() + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float* dvalue_t2 = dinp.data() + b * T * C3 + t2 * C3 + h * hs + C*2;
                    for (int i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
                        datt_bth[t2] += value_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                for (int t2 = 0; t2 <= t; t2++) {
                    for (int t3 = 0; t3 <= t; t3++) {
                        float indicator = t2 == t3 ? 1.0f : 0.0f;
                        float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += scale * local_derivative * datt_bth[t2];
                    }
                }

                // backward pass 1, the query @ key matmul
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = inp.data() + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    float* dkey_t2 = dinp.data() + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    for (int i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // preatt_bth[t2] += query_t[i] * key_t2[i]
                        // so now we have:
                        dquery_t[i] += key_t2[i] * dpreatt_bth[t2];
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2];
                    }
                }
            }
        }
    }
    return {dinp, datt, dpreatt};
}

TensorImpl TensorOpsCPU::layernorm_forward(TensorImpl& a, TensorImpl& mean, TensorImpl& rstd,
                        const TensorImpl& weight, const TensorImpl& bias, float eps) {
    int B = a.shape()[0];
    int T = a.shape()[1];
    int C = a.shape()[2];
    TensorImpl ret = TensorImpl::zerosLike(a);
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            const float* x = a.data<float>() + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = ret.data<float>() + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalized output
                float o =  n * weight.data()[i] + bias.data()[i] ; // scale and shift it
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean.data_[b * T + t] = m;
            rstd.data_[b * T + t] = s;
        }
    }
    return ret;
}

std::vector<TensorImpl> TensorOpsCPU::layernorm_backward(const TensorImpl& dout, const TensorImpl& inp,
                                                         const TensorImpl& weight,
                                                        TensorImpl& mean, TensorImpl& rstd) {
    TensorImpl dinp = TensorImpl::zerosLike(inp);
    TensorImpl dweight = TensorImpl::zeros(weight.shape());
    TensorImpl dbias = TensorImpl::zeros(weight.shape());
    int B = inp.shape()[0];
    int T = inp.shape()[1];
    int C = inp.shape()[2];
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* dout_bt = dout.data<float>() + b * T * C + t * C;
            const float* inp_bt = inp.data<float>() + b * T * C + t * C;
            float* dinp_bt = dinp.data<float>() + b * T * C + t * C;
            const float mean_bt = mean.data<float>()[b * T + t];
            const float rstd_bt = rstd.data<float>()[b * T + t];

            // first: two reduce operations
            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight.data<float>()[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight.data<float>()[i] * dout_bt[i];
                // gradient contribution to bias
                dbias.data<float>()[i] += dout_bt[i];
                // gradient contribution to weight
                dweight.data<float>()[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                float dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
    return {dinp, dweight, dbias};
}

std::pair<TensorImpl, TensorImpl> TensorOpsCPU::from_mask(const TensorImpl& a,
                                                          const TensorImpl& b) {

  const int stride = b.numel();
  const int n = a.numel();
  assert(b.shape().size() <= a.shape().size());
  TensorImpl mask;
  if (a.shape() != b.shape()){
    for (int i = 0; i < a.shape().size(); ++i) {
      int dim_mask = (i < a.shape().size() - b.shape().size())
                         ? 1 : b.shape()[i - (a.shape().size() - b.shape().size())];
      int dim_target = a.shape()[i];
      if (dim_mask != 1 && dim_mask != dim_target) {
        assert(true);
      }
    }
    mask = TensorImpl::zerosLike(a,a.device(),a.type());
    broadcastImpl<OpCpuAssign>(mask, a, b);
  }else{
    mask = b;
  }

  std::vector<int32_t> out_indices;
  int32_t count = 0;
  for (size_t i = 0; i < mask.numel(); ++i) {
    if (mask.data()[i] != 0.0f) ++count;
  }
  TensorImpl result = TensorImpl::shape({count}, a.device_, a.type_);
  out_indices.clear();
  out_indices.reserve(count);
  int32_t index = 0;

  std::vector<float> indices;
  indices.resize(count);
  for (int32_t i = 0; i < mask.numel(); ++i) {
    if (mask.data()[i] != 0.0f) {
      result.data_[index] = a.data_[i];
      indices[index] = static_cast<float>(i);
      ++index;
    }
  }
  TensorImpl indices_t =  TensorImpl(indices);
  return {result, indices_t};
}

TensorImpl TensorOpsCPU::from_mask_backward(const TensorImpl& grad,
                                      const TensorImpl& indices,
                                     const std::vector<int32_t>& a_shape) {
    TensorImpl grad_a = TensorImpl::zeros(a_shape, grad.device_, grad.type_);
    for (size_t i = 0; i < indices.numel(); ++i) {
        grad_a.data_[static_cast<int>(indices.data_[i])] = grad.data_[i];
    }
    return grad_a;
}

TensorImpl TensorOpsCPU::from_slice(const TensorImpl& a, std::vector<int> starts, std::vector<int> ends) {
    int32_t ndim = a.shape().size();
    // Step 1: Compute new shape
    std::vector<int> new_shape(ndim);
    for (int i = 0; i < ndim; ++i) {
        new_shape[i] = ends[i] - starts[i];
    }
    // Step 2: Create new tensor
    TensorImpl result = TensorImpl::shape(new_shape);

    // Step 3: Allocate memory for new tensor
    int new_size = result.numel();

    // Step 4: Copy data using linear indexing
    int total_elements = new_size;
    for (int dst_idx = 0; dst_idx < total_elements; ++dst_idx) {
        // Convert linear index to multi-dimensional indices for new tensor
        std::vector<int> new_indices(ndim);
        int temp_idx = dst_idx;
        for (int dim = 0; dim < ndim; ++dim) {
            new_indices[dim] = temp_idx / result.strides_[dim];
            temp_idx %= result.strides_[dim];
        }

        // Calculate corresponding source index in original tensor
        int src_offset = 0;
        for (int dim = 0; dim < ndim; ++dim) {
            src_offset += (new_indices[dim] + starts[dim]) * a.strides_[dim];
        }

        // Copy data
        result.data_[dst_idx] = a.data_[src_offset];
    }

    return result;

}

void TensorOpsCPU::from_slice_backward(TensorImpl &ret, const TensorImpl &b, std::vector<int> starts, std::vector<int> ends){
    int ndim = ret.shape().size();
    // Step 1: Compute new shape
    // Step 2: Create new tensor
    // Step 3: Allocate memory for new tensor
    int new_size = b.numel();

    // Step 4: Copy data using linear indexing
    int total_elements = new_size;
    for (int dst_idx = 0; dst_idx < total_elements; ++dst_idx) {
        // Convert linear index to multi-dimensional indices for new tensor
        std::vector<int> new_indices(ndim);
        int temp_idx = dst_idx;
        for (int dim = 0; dim < ndim; ++dim) {
            new_indices[dim] = temp_idx / b.strides_[dim];
            temp_idx %= b.strides_[dim];
        }
        // Calculate corresponding source index in original tensor
        int src_offset = 0;
        for (int dim = 0; dim < ndim; ++dim) {
            src_offset += (new_indices[dim] + starts[dim]) * ret.strides_[dim];
        }
        // Copy data
        ret.data_[src_offset] = b.data_[dst_idx];
    }
}

TensorImpl TensorOpsCPU::add(const TensorImpl& a, const TensorImpl& b) {
  if (a.dimCount_ == 0) {
    return add(a.data_[0], b);
  }
  if (b.dimCount_ == 0) {
    return add(a, b.data_[0]);
  }
  return opPairBroadcast<OpCpuAdd>(a, b);
}

TensorImpl TensorOpsCPU::sub(const TensorImpl& a, const TensorImpl& b) {
  if (a.dimCount_ == 0) {
    return sub(a.data_[0], b);
  }
  if (b.dimCount_ == 0) {
    return sub(a, b.data_[0]);
  }
  return opPairBroadcast<OpCpuSub>(a, b);
}

TensorImpl TensorOpsCPU::mul(const TensorImpl& a, const TensorImpl& b) {
  if (a.dimCount_ == 0) {
    return mul(a.data_[0], b);
  }
  if (b.dimCount_ == 0) {
    return mul(a, b.data_[0]);
  }
  return opPairBroadcast<OpCpuMul>(a, b);
}

TensorImpl TensorOpsCPU::div(const TensorImpl& a, const TensorImpl& b) {
  if (a.dimCount_ == 0) {
    return div(a.data_[0], b);
  }
  if (b.dimCount_ == 0) {
    return div(a, b.data_[0]);
  }
  return opPairBroadcast<OpCpuDiv>(a, b);
}

TensorImpl TensorOpsCPU::pow(const TensorImpl& a, const TensorImpl& b) {
  if (b.dimCount_ == 0) {
    return pow(a, b.data_[0]);
  }
  return opPairBroadcast<OpCpuPow>(a, b);
}

TensorImpl TensorOpsCPU::add(const TensorImpl& a, const float& b) {
  return opPair<OpCpuAdd>(a, b);
}

TensorImpl TensorOpsCPU::sub(const TensorImpl& a, const float& b) {
  return opPair<OpCpuSub>(a, b);
}

TensorImpl TensorOpsCPU::mul(const TensorImpl& a, const float& b) {
  return opPair<OpCpuMul>(a, b);
}

TensorImpl TensorOpsCPU::div(const TensorImpl& a, const float& b) {
  return opPair<OpCpuDiv>(a, b);
}

TensorImpl TensorOpsCPU::pow(const TensorImpl& a, const float& b) {
  return opPair<OpCpuPow>(a, b);
}

TensorImpl TensorOpsCPU::add(const float& a, const TensorImpl& b) {
  return opPair<OpCpuAdd>(a, b);
}

TensorImpl TensorOpsCPU::sub(const float& a, const TensorImpl& b) {
  return opPair<OpCpuSub>(a, b);
}

TensorImpl TensorOpsCPU::mul(const float& a, const TensorImpl& b) {
  return opPair<OpCpuMul>(a, b);
}

TensorImpl TensorOpsCPU::div(const float& a, const TensorImpl& b) {
  return opPair<OpCpuDiv>(a, b);
}

void TensorOpsCPU::add_(TensorImpl& a, const TensorImpl& b) {
  if (a.dimCount_ == 0) {
    a = add(a.data_[0], b);
    return;
  }
  if (b.dimCount_ == 0) {
    add_(a, b.data_[0]);
    return;
  }
  opPairBroadcast_<OpCpuAdd>(a, b);
}

void TensorOpsCPU::sub_(TensorImpl& a, const TensorImpl& b) {
  if (a.dimCount_ == 0) {
    a = sub(a.data_[0], b);
    return;
  }
  if (b.dimCount_ == 0) {
    sub_(a, b.data_[0]);
    return;
  }
  opPairBroadcast_<OpCpuSub>(a, b);
}

void TensorOpsCPU::mul_(TensorImpl& a, const TensorImpl& b) {
  if (a.dimCount_ == 0) {
    a = mul(a.data_[0], b);
    return;
  }
  if (b.dimCount_ == 0) {
    mul_(a, b.data_[0]);
    return;
  }
  opPairBroadcast_<OpCpuMul>(a, b);
}

void TensorOpsCPU::div_(TensorImpl& a, const TensorImpl& b) {
  if (a.dimCount_ == 0) {
    a = div(a.data_[0], b);
    return;
  }
  if (b.dimCount_ == 0) {
    div_(a, b.data_[0]);
    return;
  }
  opPairBroadcast_<OpCpuDiv>(a, b);
}

void TensorOpsCPU::add_(TensorImpl& a, const float& b) {
  opPair_<OpCpuAdd>(a, b);
}

void TensorOpsCPU::sub_(TensorImpl& a, const float& b) {
  opPair_<OpCpuSub>(a, b);
}

void TensorOpsCPU::mul_(TensorImpl& a, const float& b) {
  opPair_<OpCpuMul>(a, b);
}

void TensorOpsCPU::div_(TensorImpl& a, const float& b) {
  opPair_<OpCpuDiv>(a, b);
}

TensorImpl TensorOpsCPU::eq(const TensorImpl& a, const TensorImpl& b) {
  if (b.dimCount_ == 0) {
    return eq(a, b.data_[0]);
  }
  return opPairBroadcast<OpCpuEq>(a, b);
}

TensorImpl TensorOpsCPU::ne(const TensorImpl& a, const TensorImpl& b) {
  if (b.dimCount_ == 0) {
    return ne(a, b.data_[0]);
  }
  return opPairBroadcast<OpCpuNe>(a, b);
}

TensorImpl TensorOpsCPU::ge(const TensorImpl& a, const TensorImpl& b) {
  if (b.dimCount_ == 0) {
    return ge(a, b.data_[0]);
  }
  return opPairBroadcast<OpCpuGe>(a, b);
}

TensorImpl TensorOpsCPU::gt(const TensorImpl& a, const TensorImpl& b) {
  if (b.dimCount_ == 0) {
    return gt(a, b.data_[0]);
  }
  return opPairBroadcast<OpCpuGt>(a, b);
}

TensorImpl TensorOpsCPU::le(const TensorImpl& a, const TensorImpl& b) {
  if (b.dimCount_ == 0) {
    return le(a, b.data_[0]);
  }
  return opPairBroadcast<OpCpuLe>(a, b);
}

TensorImpl TensorOpsCPU::lt(const TensorImpl& a, const TensorImpl& b) {
  if (b.dimCount_ == 0) {
    return lt(a, b.data_[0]);
  }
  return opPairBroadcast<OpCpuLt>(a, b);
}

TensorImpl TensorOpsCPU::maximum(const TensorImpl& a, const TensorImpl& b) {
  if (b.dimCount_ == 0) {
    return maximum(a, b.data_[0]);
  }
  return opPairBroadcast<OpCpuMax>(a, b);
}

TensorImpl TensorOpsCPU::minimum(const TensorImpl& a, const TensorImpl& b) {
  if (b.dimCount_ == 0) {
    return minimum(a, b.data_[0]);
  }
  return opPairBroadcast<OpCpuMin>(a, b);
}

TensorImpl TensorOpsCPU::eq(const TensorImpl& a, const float& b) {
  return opPair<OpCpuEq>(a, b);
}

TensorImpl TensorOpsCPU::ne(const TensorImpl& a, const float& b) {
  return opPair<OpCpuNe>(a, b);
}

TensorImpl TensorOpsCPU::ge(const TensorImpl& a, const float& b) {
  return opPair<OpCpuGe>(a, b);
}

TensorImpl TensorOpsCPU::gt(const TensorImpl& a, const float& b) {
  return opPair<OpCpuGt>(a, b);
}

TensorImpl TensorOpsCPU::le(const TensorImpl& a, const float& b) {
  return opPair<OpCpuLe>(a, b);
}

TensorImpl TensorOpsCPU::lt(const TensorImpl& a, const float& b) {
  return opPair<OpCpuLt>(a, b);
}

TensorImpl TensorOpsCPU::maximum(const TensorImpl& a, const float& b) {
  return opPair<OpCpuMax>(a, b);
}

TensorImpl TensorOpsCPU::minimum(const TensorImpl& a, const float& b) {
  return opPair<OpCpuMin>(a, b);
}

void TensorOpsCPU::abs_(TensorImpl& t) { opSingle_<OpCpuAbs_>(t); }

void TensorOpsCPU::sin_(TensorImpl& t) { opSingle_<OpCpuSin_>(t); }

void TensorOpsCPU::cos_(TensorImpl& t) { opSingle_<OpCpuCos_>(t); }

void TensorOpsCPU::sqrt_(TensorImpl& t) { opSingle_<OpCpuSqrt_>(t); }

void TensorOpsCPU::tanh_(TensorImpl& t) { opSingle_<OpCpuTanh_>(t); }

void TensorOpsCPU::exp_(TensorImpl& t) { opSingle_<OpCpuExp_>(t); }

void TensorOpsCPU::log_(TensorImpl& t) { opSingle_<OpCpuLog_>(t); }

TensorImpl TensorOpsCPU::abs(const TensorImpl& t) {
  return opSingle<OpCpuAbs>(t);
}

TensorImpl TensorOpsCPU::sin(const TensorImpl& t) {
  return opSingle<OpCpuSin>(t);
}

TensorImpl TensorOpsCPU::cos(const TensorImpl& t) {
  return opSingle<OpCpuCos>(t);
}

TensorImpl TensorOpsCPU::sqrt(const TensorImpl& t) {
  return opSingle<OpCpuSqrt>(t);
}

TensorImpl TensorOpsCPU::tanh(const TensorImpl& t) {
  return opSingle<OpCpuTanh>(t);
}

TensorImpl TensorOpsCPU::exp(const TensorImpl& t) {
  return opSingle<OpCpuExp>(t);
}

TensorImpl TensorOpsCPU::log(const TensorImpl& t) {
  return opSingle<OpCpuLog>(t);
}

void TensorOpsCPU::clampMin_(TensorImpl& t, float min) {
  opPair_<OpCpuMax>(t, min);
}

void TensorOpsCPU::clampMax_(TensorImpl& t, float max) {
  opPair_<OpCpuMin>(t, max);
}

void TensorOpsCPU::clamp_(TensorImpl& t, float min, float max) {
  for (int32_t i = 0; i < t.elemCount_; i++) {
    t.data_[i] = std::max(min, std::min(t.data_[i], max));
  }
}

TensorImpl TensorOpsCPU::clampMin(const TensorImpl& t, float min) {
  return opPair<OpCpuMax>(t, min);
}

TensorImpl TensorOpsCPU::clampMax(const TensorImpl& t, float max) {
  return opPair<OpCpuMin>(t, max);
}

TensorImpl TensorOpsCPU::clamp(const TensorImpl& t, float min, float max) {
  auto result = TensorImpl::shape(t.shape(), t.device_);
  for (int32_t i = 0; i < t.elemCount_; i++) {
    result.data_[i] = std::max(min, std::min(t.data_[i], max));
  }
  return result;
}

TensorImpl TensorOpsCPU::min(const TensorImpl& t) {
  if (t.dimCount_ == 0) {
    return t;
  }
  auto ret = TensorImpl::scalar(t.device_);
  reduceAll<OpCpuReduceMin>(ret.data_, t.data_, t.elemCount_);
  return ret;
}

TensorImpl TensorOpsCPU::max(const TensorImpl& t) {
  if (t.dimCount_ == 0) {
    return t;
  }
  auto ret = TensorImpl::scalar(t.device_);
  reduceAll<OpCpuReduceMax>(ret.data_, t.data_, t.elemCount_);
  return ret;
}

TensorImpl TensorOpsCPU::sum(const TensorImpl& t) {
  if (t.dimCount_ == 0) {
    return t;
  }
  auto ret = TensorImpl::scalar(t.device_);
  reduceAll<OpCpuReduceSum>(ret.data_, t.data_, t.elemCount_);
  return ret;
}

TensorImpl TensorOpsCPU::mean(const TensorImpl& t) {
  if (t.dimCount_ == 0) {
    return t;
  }
  auto ret = TensorImpl::scalar(t.device_);
  reduceAll<OpCpuReduceSum>(ret.data_, t.data_, t.elemCount_);
  const auto r = 1.f / static_cast<float>(t.elemCount_);
  ret.data_[0] *= r;
  return ret;
}

TensorImpl TensorOpsCPU::var(const TensorImpl& t, bool unbiased) {
  if (t.dimCount_ == 0) {
    return TensorImpl::scalar(0, t.device_);
  }
  const auto meanVal = mean(t);
  auto squaredDiff = 0.f;
  for (int32_t i = 0; i < t.elemCount_; i++) {
    const auto diff = t.data_[i] - meanVal.data_[0];
    squaredDiff += diff * diff;
  }
  auto ret = TensorImpl::scalar(squaredDiff, t.device_);
  const auto n = static_cast<float>(t.elemCount_);
  auto r = 1.f / n;
  if (unbiased) {
    r *= n / (n - 1.f);
  }
  ret.data_[0] *= r;
  return ret;
}

TensorImpl TensorOpsCPU::argmin(const TensorImpl& t) {
  if (t.dimCount_ == 0) {
    return TensorImpl::scalar(0, t.device_);
  }
  auto ret = TensorImpl::scalar(t.device_);
  reduceAllIdx<OpCpuReduceMin>(ret.data_, t.data_, t.elemCount_);
  return ret;
}

TensorImpl TensorOpsCPU::argmax(const TensorImpl& t) {
  if (t.dimCount_ == 0) {
    return TensorImpl::scalar(0, t.device_);
  }
  auto ret = TensorImpl::scalar(t.device_);
  reduceAllIdx<OpCpuReduceMax>(ret.data_, t.data_, t.elemCount_);
  return ret;
}

std::pair<TensorImpl, TensorImpl> TensorOpsCPU::min(const TensorImpl& t,
                                                    int32_t dim,
                                                    bool keepDims) {
  if (t.dimCount_ == 0) {
    return {t, TensorImpl::scalar(0, t.device_)};
  }
  return reduceDim(t, dim, keepDims, std::numeric_limits<float>::max(),
                   std::less<>());
}

std::pair<TensorImpl, TensorImpl> TensorOpsCPU::max(const TensorImpl& t,
                                                    int32_t dim,
                                                    bool keepDims) {
  if (t.dimCount_ == 0) {
    return {t, TensorImpl::scalar(0, t.device_)};
  }
  return reduceDim(t, dim, keepDims, -std::numeric_limits<float>::max(),
                   std::greater<>());
}

TensorImpl TensorOpsCPU::sum(const TensorImpl& t,
                             const std::vector<int32_t>& dims, bool keepDims) {
  if (t.dimCount_ == 0) {
    return t;
  }
  return reduceMultiDim(
      t, dims, keepDims,
      [](TensorImpl& ret, const TensorImpl& t, int32_t retIdx, int32_t srcIdx) {
        ret.data_[retIdx] += t.data_[srcIdx];
      });
}

TensorImpl TensorOpsCPU::mean(const TensorImpl& t,
                              const std::vector<int32_t>& dims, bool keepDims) {
  if (t.dimCount_ == 0) {
    return t;
  }
  auto ret = sum(t, dims, keepDims);
  if (!ret.empty()) {
    auto reduceSize = (float)t.elemCount_ / (float)ret.elemCount_;
    auto r = 1.f / reduceSize;
    mul_(ret, r);
  }
  return ret;
}

TensorImpl TensorOpsCPU::triangle(const TensorImpl& t, int32_t diagonal,
                                   bool lower) {
  auto ret = TensorImpl::shape(t.shape_, t.device_);
  const auto rows = t.shape_[0];
  const auto cols = t.shape_[1];

  int32_t idx = 0;
  for (auto i = 0; i < rows; i++) {
    idx = i * cols;
    for (auto j = 0; j < cols; j++) {
      if ((lower && j <= i + diagonal) || (!lower && j >= i + diagonal)) {
        ret.data_[idx] = t.data_[idx];
      } else {
        ret.data_[idx] = 0.0f;
      }
      idx++;
    }
  }
  return ret;
}

TensorImpl TensorOpsCPU::var(const TensorImpl& t,
                             const std::vector<int32_t>& dims, bool unbiased,
                             bool keepDims) {
  if (t.dimCount_ == 0) {
    return TensorImpl::scalar(0, t.device_);
  }
  auto meanTensor = mean(t, dims, true);
  auto ret = reduceMultiDim(t, dims, keepDims,
                            [&meanTensor](TensorImpl& ret, const TensorImpl& t,
                                          int32_t retIdx, int32_t srcIdx) {
                              float diff =
                                  t.data_[srcIdx] - meanTensor.data_[retIdx];
                              ret.data_[retIdx] += diff * diff;
                            });
  if (!ret.empty()) {
    auto reduceSize = (float)t.elemCount_ / (float)ret.elemCount_;
    auto r = 1.f / reduceSize;
    if (unbiased) {
      r *= reduceSize / (reduceSize - 1.f);
    }
    mul_(ret, r);
  }
  return ret;
}

TensorImpl TensorOpsCPU::permute(const TensorImpl& t,
                                 const std::vector<int32_t>& dims) {
  auto retShape = t.shape_;
  reorderIndices(retShape.data(), dims);
  auto ret = TensorImpl::shape(retShape, t.device_);

  for (int32_t i = 0; i < t.elemCount_; i++) {
    int32_t originIndex = 0;
    int32_t offset = i;
    for (int32_t d = 0; d < t.dimCount_; d++) {
      originIndex += (offset / ret.strides_[d]) * t.strides_[dims[d]];
      offset %= ret.strides_[d];
    }
    ret.data_[i] = t.data_[originIndex];
  }
  return ret;
}

TensorImpl TensorOpsCPU::transpose2D(const TensorImpl& t) {
  return permute(t, {1, 0});
}

TensorImpl TensorOpsCPU::index(
    const TensorImpl& t,
    const std::vector<std::reference_wrapper<TensorImpl>>& indices) {
  auto len = (int32_t)indices.size();
  auto fistDim = (int32_t)indices[0].get().elemCount_;
  auto dimStride = t.strides_[len - 1];
  Shape retShape = {fistDim};
  for (auto i = len; i < t.dimCount_; i++) {
    retShape.push_back(t.shape_[i]);
  }
  auto retTensor = TensorImpl::shape(retShape, t.device_);

  static int32_t subIndices[TENSOR_MAX_DIMS];
  for (int32_t i = 0; i < fistDim; i++) {
    getSubIndices(subIndices, t, indices, i);
    int32_t dataIdx = indicesToOffset(t.strides_, subIndices);
    copyOnDevice(&retTensor.data_[dimStride * i], &t.data_[dataIdx],
                 dimStride * sizeof(float));
  }

  return retTensor;
}

void TensorOpsCPU::indexPut_(
    TensorImpl& t,
    const std::vector<std::reference_wrapper<TensorImpl>>& indices, float val) {
  auto len = (int32_t)indices.size();
  auto fistDim = (int32_t)indices[0].get().elemCount_;
  auto dimStride = t.strides_[len - 1];

  static int32_t subIndices[TENSOR_MAX_DIMS];
  for (int32_t i = 0; i < fistDim; i++) {
    getSubIndices(subIndices, t, indices, i);
    int32_t dataIdx = indicesToOffset(t.strides_, subIndices);
    fillConstant_(&t.data_[dataIdx], val, dimStride);
  }
}

void TensorOpsCPU::indexPut_(
    TensorImpl& t,
    const std::vector<std::reference_wrapper<TensorImpl>>& indices,
    const TensorImpl& val) {
  auto len = (int32_t)indices.size();
  auto fistDim = (int32_t)indices[0].get().elemCount_;
  auto dimStride = t.strides_[len - 1];
  assert(val.elemCount_ == dimStride * fistDim);

  static int32_t subIndices[TENSOR_MAX_DIMS];
  for (int32_t i = 0; i < fistDim; i++) {
    getSubIndices(subIndices, t, indices, i);
    int32_t dataIdx = indicesToOffset(t.strides_, subIndices);
    copyOnDevice(&t.data_[dataIdx], &val.data_[dimStride * i],
                 dimStride * sizeof(float));
  }
}
TensorImpl TensorOpsCPU::im2col1D(const TensorImpl& t,
                                Size1D kernel,
                                Size1D stride,
                                Size1D padding) {
    assert(t.dimCount_ == 2 || t.dimCount_ == 3);
    const int32_t batch = (t.dimCount_ == 3) ? t.shape_[0] : 1;
    const int32_t channels = (t.dimCount_ == 3) ? t.shape_[1] : t.shape_[0];
    const int32_t length = (t.dimCount_ == 3) ? t.shape_[2] : t.shape_[1];
    const int32_t outLength = (length - kernel.d + 2 * padding.d) / stride.d + 1;

    const int32_t colH = outLength;
    const int32_t colW = channels * kernel.d;
    auto retTensor = TensorImpl::shape({batch * colH, colW}, t.device_);

    const int32_t imStride = (t.dimCount_ == 3) ? t.strides_[2] : t.strides_[1];

    for (int32_t n = 0; n < batch; ++n) {
        for (int32_t c = 0; c < channels; ++c) {
            for (int32_t kl = 0; kl < kernel.d; ++kl) {
                for (int32_t l = 0; l < outLength; ++l) {

                    const int32_t imPos = l * stride.d + kl - padding.d;

                    const int32_t colIdx = n * outLength + l;
                    const int32_t colWIdx = c * kernel.d + kl;

                    if (imPos < 0 || imPos >= length) {
                        retTensor.data_[colIdx * colW + colWIdx] = 0;
                    } else {

                        const int32_t imOffset = n * (channels * imStride)
                                               + c * imStride
                                               + imPos;
                        retTensor.data_[colIdx * colW + colWIdx] = t.data_[imOffset];
                    }
                }
            }
        }
    }

    return retTensor;
}
TensorImpl TensorOpsCPU::im2col(const TensorImpl& t, Size2D kernel,
                                Size2D stride, Size2D padding) {
  // this: [C, H, W], [N, C, H, W]
  assert(t.dimCount_ == 3 || t.dimCount_ == 4);
  int32_t batch = (t.dimCount_ == 4) ? t.shape_[0] : 1;
  int32_t channels = (t.dimCount_ == 4) ? t.shape_[1] : t.shape_[0];
  int32_t height = (t.dimCount_ == 4) ? t.shape_[2] : t.shape_[1];
  int32_t width = (t.dimCount_ == 4) ? t.shape_[3] : t.shape_[2];
  int32_t outH = (height - kernel.h + 2 * padding.h) / stride.h + 1;
  int32_t outW = (width - kernel.w + 2 * padding.w) / stride.w + 1;

  int32_t colH = outH * outW;
  int32_t colW = channels * kernel.h * kernel.w;
  auto retTensor = TensorImpl::shape({batch * colH, colW}, t.device_);

  int32_t imStride = t.strides_[0];
  for (int32_t n = 0; n < batch; n++) {
    for (int32_t c = 0; c < channels; c++) {
      for (int32_t kh = 0; kh < kernel.h; kh++) {
        for (int32_t kw = 0; kw < kernel.w; kw++) {
          for (int32_t h = 0; h < outH; h++) {
            for (int32_t w = 0; w < outW; w++) {
              int32_t imRow = h * stride.h + kh - padding.h;
              int32_t imCol = w * stride.w + kw - padding.w;
              int32_t colIdx = (n * outH + h) * outW + w;
              int32_t colWIdx = c * kernel.h * kernel.w + kh * kernel.w + kw;
              if (imRow < 0 || imRow >= height || imCol < 0 || imCol >= width) {
                retTensor.data_[colIdx * colW + colWIdx] = 0;  // zero padding
              } else {
                int32_t imgIdx = imCol + width * (imRow + height * c);
                retTensor.data_[colIdx * colW + colWIdx] =
                    t.data_[n * imStride + imgIdx];
              }
            }
          }
        }
      }
    }
  }
  return retTensor;
}

TensorImpl TensorOpsCPU::col2im(const TensorImpl& t, const Shape& shape,
                                Size2D kernel, Size2D stride, Size2D padding) {
  // shape: [C, H, W], [N, C, H, W]
  assert(shape.size() == 3 || shape.size() == 4);
  int32_t batch = (shape.size() == 4) ? shape[0] : 1;
  int32_t channels = (shape.size() == 4) ? shape[1] : shape[0];
  int32_t height = (shape.size() == 4) ? shape[2] : shape[1];
  int32_t width = (shape.size() == 4) ? shape[3] : shape[2];

  auto outH = (height - kernel.h + 2 * padding.h) / stride.h + 1;
  auto outW = (width - kernel.w + 2 * padding.w) / stride.w + 1;

  // int32_t colH = outH * outW;
  int32_t colW = channels * kernel.h * kernel.w;

  auto retTensor = TensorImpl::zeros(shape, t.device_);

  auto imStride = retTensor.strides_[0];
  for (int32_t n = 0; n < batch; n++) {
    for (int32_t c = 0; c < channels; c++) {
      for (int32_t kh = 0; kh < kernel.h; kh++) {
        for (int32_t kw = 0; kw < kernel.w; kw++) {
          for (int32_t h = 0; h < outH; h++) {
            for (int32_t w = 0; w < outW; w++) {
              int32_t imRow = h * stride.h + kh - padding.h;
              int32_t imCol = w * stride.w + kw - padding.w;
              int32_t colIdx = (n * outH + h) * outW + w;
              int32_t colWIdx = c * kernel.h * kernel.w + kh * kernel.w + kw;
              if (imRow >= 0 && imRow < height && imCol >= 0 && imCol < width) {
                int32_t imgIdx = imCol + width * (imRow + height * c);
                retTensor.data_[n * imStride + imgIdx] +=
                    t.data_[colIdx * colW + colWIdx];
              }
            }
          }
        }
      }
    }
  }
  return retTensor;
}

TensorImpl TensorOpsCPU::col2im1D(const TensorImpl& t,
                                const Shape& shape,
                                Size1D kernel,
                                Size1D stride,
                                Size1D padding) {

    assert(shape.size() == 2 || shape.size() == 3);

    const int32_t batch = (shape.size() == 3) ? shape[0] : 1;
    const int32_t channels = (shape.size() == 3) ? shape[1] : shape[0];
    const int32_t length = (shape.size() == 3) ? shape[2] : shape[1];

    const int32_t outLength = (length - kernel.d + 2 * padding.d) / stride.d + 1;

    const int32_t colW = channels * kernel.d;

    auto retTensor = TensorImpl::zeros(shape, t.device_);

    const int32_t stride_n = (shape.size() == 3) ? retTensor.strides_[0] : 1;
    const int32_t stride_c = (shape.size() == 3) ? retTensor.strides_[1] : retTensor.strides_[0];
    const int32_t stride_l = (shape.size() == 3) ? retTensor.strides_[2] : retTensor.strides_[1];

    for (int32_t n = 0; n < batch; ++n) {
        for (int32_t l = 0; l < outLength; ++l) {
            int32_t start = l * stride.d - padding.d;
            int32_t end = start + kernel.d - 1;
            if (end < 0 || start >= length) continue;

            const int32_t colIdx = n * outLength + l;
            for (int32_t c = 0; c < channels; ++c) {
                for (int32_t kl = 0; kl < kernel.d; ++kl) {
                    int32_t imPos = start + kl;

                    if (imPos >= 0 && imPos < length) {
                        int32_t imOffset = n * stride_n + c * stride_c + imPos * stride_l;
                        int32_t colWIdx = c * kernel.d + kl;
                        retTensor.data_[imOffset] += t.data_[colIdx * colW + colWIdx];
                    }
                }
            }
        }
    }

    return retTensor;
}

TensorImpl TensorOpsCPU::dot(const TensorImpl& a, const TensorImpl& b) {
  float ret = 0.f;
  for (int32_t i = 0; i < a.elemCount_; i++) {
    ret += a.data_[i] * b.data_[i];
  }
  return TensorImpl::scalar(ret, a.device_);
}


void TensorOpsCPU::gemm(float* c, const float * a, const float * b, int32_t m,
                        int32_t k, int32_t n, bool transA, bool transB,Dtype Ta, Dtype Tc) {

#ifdef USE_BLAS
  cblas_sgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans,
              transB ? CblasTrans : CblasNoTrans, m, n, k, 1.f, a,
              transA ? m : k, b, transB ? k : n, 0.f, c, n);
  return;
#endif
  for (int i = 0; i < m * n; i++) {
    c[i] = 0.0f;
  }

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int p = 0; p < k; p++) {
        float aVal = transA ? a[p * m + i] : a[i * k + p];
        float bVal = transB ? b[j * k + p] : b[p * n + j];
        c[i * n + j] += aVal * bVal;
      }
    }
  }
}


std::pair<TensorImpl, TensorImpl> TensorOpsCPU::leakyrelu(const TensorImpl& a, float rate) {
    TensorImpl ret = TensorImpl::shape(a.shape_, a.device_, a.type_);
    // in cpu, we didn't apply other type to data, only float32
    TensorImpl mask = TensorImpl::shape(a.shape_, a.device_, Dtype::float32);
    int N = a.numel();
    for (int i = 0; i < N; ++i) {
        bool condition = a.data_[i] > 0.0f;
        ret.data_[i] = condition ? a.data_[i] : a.data_[i] * rate;
        mask.data_[i] = condition;
    }
    return {ret, mask};
}

TensorImpl TensorOpsCPU::leakyrelu_backward(const TensorImpl&
        a, const TensorImpl& mask, float rate) {
    TensorImpl ret = TensorImpl::shape(a.shape_, a.device_, a.type_);
    int N = a.numel();
    for (int i = 0; i < N; ++i) {
        float mask_float = mask.data_[i] ? 1.0f : rate;
        ret.data_[i] = a.data_[i] * mask_float;
    }
    return ret;
}

TensorImpl TensorOpsCPU::concat(const TensorImpl& a , const TensorImpl& b, int32_t dim_){
  Shape shape1 = a.shape();
  Shape shape2 = b.shape();
  std::vector<int32_t> output_shape = shape1;
  output_shape[dim_] += shape2[dim_];
  TensorImpl output = TensorImpl::zeros(output_shape);
  const size_t elem_size = sizeof(int32_t);
  auto* out_ptr = output.data();
  const auto* in1_ptr = a.data();
  const auto* in2_ptr = b.data();
  const size_t outer_size = std::accumulate(shape1.begin(), shape1.begin() + dim_,
                                            1, std::multiplies<int32_t>());
  const size_t inner_size = std::accumulate(shape1.begin() + dim_ + 1, shape1.end(),
                                            1, std::multiplies<int32_t>());
  const size_t copy_size1 = shape1[dim_] * inner_size * elem_size;
  const size_t copy_size2 = shape2[dim_] * inner_size * elem_size;

  for (size_t i = 0; i < outer_size; ++i) {

    memcpy(out_ptr, in1_ptr, copy_size1);
    in1_ptr += copy_size1 / elem_size;
    out_ptr += copy_size1 / elem_size;

    memcpy(out_ptr, in2_ptr, copy_size2);
    in2_ptr += copy_size2 / elem_size;
    out_ptr += copy_size2 / elem_size;
  }
  return output;
}

std::vector<TensorImpl> TensorOpsCPU::split(
    const TensorImpl& t,
    int32_t splitSize,
    int32_t dim,
    Shape t1_shape,
    Shape t2_shape)
{
  TensorImpl grad1, grad2;
  grad1 = TensorImpl::zeros(t1_shape);
  grad2 = TensorImpl::zeros(t2_shape);
  const size_t outer_size = std::accumulate( t.shape_.begin(),  t.shape_.begin() + dim, 1, std::multiplies<int32_t>());
  const size_t inner_size = std::accumulate( t.shape_.begin() + dim + 1, t.shape_.end(), 1, std::multiplies<int32_t>());
  const size_t elem_size = sizeof(int32_t);
  auto grad_data = t.data();
  auto grad1_data = grad1.data();
  auto grad2_data =  grad2.data();
  int a = grad1.shape_[dim];
  int b = grad2.shape_[dim];
  for (size_t i = 0; i < outer_size; ++i) {
    const auto src = grad_data + i * t.shape_[dim] * inner_size;
      auto dest1 = grad1_data + i * a * inner_size;
      copyOnDevice(dest1, src, a * inner_size * elem_size);
      const auto src2 = src + a * inner_size;
      auto dest2 = grad2_data + i * b * inner_size;
      copyOnDevice(dest2, src2, b * inner_size * elem_size);
  }
  return {grad1, grad2};
}

TensorImpl TensorOpsCPU::upsample_forward(const TensorImpl& Q, int32_t scale_factor) {
  throw std::runtime_error("We have not implement in CPU yet");

}

TensorImpl TensorOpsCPU::upsample_backward(const TensorImpl& Q, int32_t scale_factor) {
    throw std::runtime_error("We have not implement in CPU yet");
}

void TensorOpsCPU::convertTypeOnDevice(void* dst, void* src, size_t count, Dtype Ti ,Dtype To) {
    throw std::runtime_error("We have not implement in CPU yet");
}

}  // namespace TinyTorch


