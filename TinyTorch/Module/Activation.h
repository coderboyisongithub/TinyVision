#pragma once
#include "Module_base.h"

namespace TinyTorch::nn {
class Relu : public Module {
 public:
    std::string name() const override { return "Relu"; }
  Tensor forward(Tensor &input) override;
};

class LeakyRelu : public Module {
 public:
    std::string name() const override { return "LeakyRelu"; }
  explicit LeakyRelu(float rate=0.1) : rate_(rate) {}
  Tensor forward(Tensor &input) override;
 private:
  float rate_;
};
}