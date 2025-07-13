#pragma once
#include "Module_base.h"

namespace TinyTorch::nn {
class Relu : public Module {
 public:
  REGISTER_MODULE_NAME(Relu)
  Tensor forward(Tensor &input) override;
};

class LeakyRelu : public Module {
 public:
  REGISTER_MODULE_NAME(LeakyRelu)
  explicit LeakyRelu(float rate=0.1) : rate_(rate) {}
  Tensor forward(Tensor &input) override;
 private:
  float rate_;
};
}