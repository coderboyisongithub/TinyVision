#pragma once
#include "Module_base.h"

namespace TinyTorch::nn {


class Softmax : public Module {
 public:
    std::string name() const override { return "Softmax"; }
  explicit Softmax(int32_t dim) : dim_(dim) {}

  Tensor forward(Tensor &x) override;

 private:
  int32_t dim_;
};

class LogSoftmax : public Module {
 public:
    std::string name() const override { return "LogSoftmax"; }
  explicit LogSoftmax(int32_t dim) : dim_(dim) {}
  Tensor forward(Tensor &x) override;
 private:
  int32_t dim_;
};

}