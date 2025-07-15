#pragma once
#include "Module_base.h"
namespace TinyTorch::nn {
class MaxPool2D : public Module {
 public:
    REGISTER_MODULE_NAME(MaxPool2D)
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
};

class AvgPool2D : public Module {
 public:
    REGISTER_MODULE_NAME(AvgPool2D)
  explicit AvgPool2D(Size2D kernelSize,
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
};

class AdaptiveAvgPool2D : public Module {
 public:
    REGISTER_MODULE_NAME(AdaptiveAvgPool2D)
  explicit AdaptiveAvgPool2D(Size2D output_size)
      : output_size_(output_size){}
  Tensor forward(Tensor &x) override;
 private:
  Size2D output_size_;
  static Size2D calculate_kernel_size(
    Shape input_size,
    Size2D output_size
    ) {
        int64_t h_in = input_size[0];
        int64_t w_in = input_size[1];
        int64_t h_out = output_size.h;
        int64_t w_out = output_size.w;
        return Size2D(
            h_in - (h_out - 1) * (h_in / h_out),
            w_in - (w_out - 1) * (w_in / w_out));
    }
};
}