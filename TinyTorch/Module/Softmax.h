#pragma once
#include "Module_base.h"

namespace TinyTorch::nn {


class Softmax : public Module {
 public:
    REGISTER_MODULE_NAME(Softmax)
  explicit Softmax(int32_t dim) : dim_(dim) {}
  Tensor forward(Tensor &x) override;
 private:
  int32_t dim_;
};

class LogSoftmax : public Module {
 public:
    REGISTER_MODULE_NAME(LogSoftmax)
  explicit LogSoftmax(int32_t dim) : dim_(dim) {}
  Tensor forward(Tensor &x) override;
 private:
  int32_t dim_;
};

}