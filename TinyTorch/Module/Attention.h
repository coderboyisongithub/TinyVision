#pragma once
#include "Module_base.h"
#include "Linear.h"

namespace TinyTorch::nn {
class MultiheadAttention : public Module {
 public:
    std::string name() const override { return "MultiheadAttention"; }
  explicit MultiheadAttention(int32_t inFeatures, int32_t head = 8,
            int is_casual = 0, bool bias = true, bool bias_proj = true);
  Tensor forward(Tensor &input) override;
  std::vector<Tensor *> parameters() override;
  std::vector<Tensor *> states() override;
  void resetParameters() override;
  void zeroGrad() override;
  nn::Linear qkv_proj(){return qkv_proj_;};
  nn::Linear last_proj(){return last_proj_;};
 private:
  int is_casual_;
  nn::Linear qkv_proj_;
  nn::Linear last_proj_;
  int32_t inFeatures_;
  int32_t head_;
  bool useBias_;
  bool useprojBias_;
};
}