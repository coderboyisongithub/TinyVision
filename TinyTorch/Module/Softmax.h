#pragma once
#include "Module_base.h"

namespace TinyTorch::nn {


class Softmax : public Module {
 public:
  explicit Softmax(int32_t dim) : dim_(dim) {}
  Tensor forward(Tensor &x) override;
 private:
  int32_t dim_;
  std::string name_ = "Softmax";
};

class LogSoftmax : public Module {
 public:
  explicit LogSoftmax(int32_t dim) : dim_(dim) {}
  Tensor forward(Tensor &x) override;
 private:
  int32_t dim_;
  std::string name_ = "LogSoftmax";
};

}