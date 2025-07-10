#pragma once
#include "Module_base.h"

namespace TinyTorch::nn {
class Relu : public Module {
 public:
  Tensor forward(Tensor &input) override;
  private:
    std::string name_ = "Relu";
};

class LeakyRelu : public Module {
 public:
  explicit LeakyRelu(float rate=0.1) : rate_(rate) {}
  Tensor forward(Tensor &input) override;
 private:
  float rate_;
  std::string name_ = "LeakyRelu";
};
}