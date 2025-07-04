#pragma once
#include "Module_base.h"
namespace TinyTorch::nn {
class Linear : public Module {

 public:
   std::string name() const override { return "Linear"; }
  Linear(int32_t inFeatures, int32_t outFeatures, bool bias = true);
  Tensor forward(Tensor &input) override;
  std::vector<Tensor *> parameters() override;
  std::vector<Tensor *> states() override;
  void resetParameters() override;
  void zeroGrad() override;
  Tensor &weights() { return weights_; }
  Tensor &bias() { return bias_; }
 private:
  int32_t inFeatures_;
  int32_t outFeatures_;
  bool useBias_;
  Tensor weights_;
  Tensor bias_;
};

}