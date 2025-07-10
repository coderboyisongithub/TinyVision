#pragma once
#include "Module_base.h"
namespace TinyTorch::nn {
class MaxPool2D : public Module {
 public:
  explicit MaxPool2D(Size2D kernelSize,
                     std::optional<Size2D> stride = std::nullopt,
                     Size2D padding = 0)
      : kernelSize_(kernelSize),
        stride_(stride.has_value() ? stride.value() : kernelSize),
        padding_(padding) {}

  Tensor forward(Tensor &x) override;

 private:
  Size2D kernelSize_;
  Size2D stride_;
  Size2D padding_;
  std::string name_ = "MaxPool2D";
};
}