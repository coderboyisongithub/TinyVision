#pragma once
#include "Module_base.h"

namespace TinyTorch::nn {
class BatchNorm2D : public Module {
 public:
  REGISTER_MODULE_NAME(BatchNorm2D)
  explicit BatchNorm2D(int32_t numFeatures, float eps = 1e-5,
                       float momentum = 0.1f, bool affine = true,
                       bool trackRunningStats = true);

  Tensor forward(Tensor &input) override;
  std::vector<Tensor *> parameters() override;
  std::vector<Tensor *> states() override;
  void resetParameters() override;
  void zeroGrad() override;

  Tensor &weights() { return weights_; }
  Tensor &bias() { return bias_; }

  Tensor &runningMean() { return runningMean_; }
  Tensor &runningVar() { return runningVar_; }

 private:
  int32_t numFeatures_;
  float eps_;
  float momentum_;
  bool affine_;
  bool trackRunningStats_;

  Tensor weights_;
  Tensor bias_;

  Tensor runningMean_;
  Tensor runningVar_;
  int32_t numBatchesTracked_;
};
}